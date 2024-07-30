from argparse import ArgumentParser
import os
from pathlib import Path

import pandas as pd
import torch
import yaml

from eeg_online.utils.seed import seed_everything
from eeg_online.utils.get_datamodule_cls import get_datamodule_cls
from eeg_online.utils.get_model_cls import get_model_cls
from eeg_online.utils.get_tta_cls import get_tta_cls
from eeg_online.utils.load_checkpoint import load_checkpoint


CONFIG_DIR = os.path.join(Path(__file__).resolve().parents[2], "configs")
DEFAULT_SOURCE_CONFIG = "lee_basenet.yaml"
DEFAULT_CONFIG = "lee_tta.yaml"


def run_online_adaptation(source_config: dict, config: dict):
    run_dir, ckpt_name, n_subjects = config.get("run_dir"), config.get("ckpt_name"), config.get("n_subjects")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessing_dict = source_config.get("preprocessing")
    _ = [preprocessing_dict.pop(key, None) for key in [
        "batch_size", "n_train_runs", "n_train_subjects"]]
    # always set false as we (eventually) do online alignment
    preprocessing_dict["alignment"] = False

    model_cls = get_model_cls(source_config["model"])
    tta_cls = get_tta_cls(config["tta_method"])
    source_dm = source_config.get("datamodule", "LargeDatasetDataModuleLMSO")
    datamodule_cls = get_datamodule_cls(source_dm[:-4] + "TTA")


    datamodule = datamodule_cls(n_subjects, n_folds=n_subjects,
                                preprocessing_dict=source_config.get("preprocessing"))
    subject_ids = datamodule.dataset.all_subject_ids

    results_df = pd.DataFrame(index=subject_ids[:n_subjects],
                              columns=[f"run_{i}" for i in datamodule.dataset.test_run_ids])

    for version, subject_id in enumerate(subject_ids[:n_subjects]):
        seed_everything(source_config.get("seed"))
        datamodule.setup_fold(version)

        # load checkpoint
        ckpt_path = load_checkpoint(run_dir, f"version_{version}", ckpt_name)
        model = model_cls.load_from_checkpoint(ckpt_path, map_location=device)

        model = tta_cls(model, config.get("tta_config"))

        # run adaptation
        outputs, labels = [], []
        with torch.no_grad():
            for batch in datamodule.predict_dataloader():
                x, y = batch
                output = torch.sigmoid(model(x.to(device)))
                outputs.append(output)
                labels.append(y)

        outputs = torch.concatenate(outputs)
        labels = torch.concatenate(labels).to(device)

        # reshape outputs
        n_windows_per_trial = datamodule.dataset.n_windows_per_trial
        n_trials = outputs.shape[0] // n_windows_per_trial
        y_pred = outputs.reshape(n_trials, n_windows_per_trial)
        y_test = labels[::n_windows_per_trial]
        taccs = ((y_pred.mean(dim=-1) > 0.5).float() == y_test).float()

        # write to dataframe
        trials_per_run = [len(datamodule.dataset.data_dict[subject_id]["labels"]
                              [f"run_{i}"]) for i in datamodule.dataset.test_run_ids]
        trials_counter = 0
        for run_idx, run_id in enumerate(results_df.columns):
            results_df.at[subject_id, run_id] = taccs[trials_counter:trials_counter +
                                                                     trials_per_run[
                                                                         run_idx]].mean().item()
            trials_counter += trials_per_run[run_idx]

    print("results per subject")
    print(results_df.mean(1))
    print("results per run")
    print(results_df.mean(0))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source_config", default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    with open(os.path.join(CONFIG_DIR, args.source_config)) as f:
        source_config = yaml.safe_load(f)
    with open(os.path.join(CONFIG_DIR, args.config)) as f:
        config = yaml.safe_load(f)

    run_online_adaptation(source_config, config)

