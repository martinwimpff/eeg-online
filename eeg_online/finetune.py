from argparse import ArgumentParser
import os
from pathlib import Path

import pandas as pd
from pytorch_lightning import Trainer
import torch
import yaml

from eeg_online.models import ClassificationModule, reset_bn

from eeg_online.utils.get_datamodule_cls import get_datamodule_cls
from eeg_online.utils.get_model_cls import get_model_cls
from eeg_online.utils.load_checkpoint import load_checkpoint
from eeg_online.utils.seed import seed_everything


CONFIG_DIR = os.path.join(Path(__file__).resolve().parents[1], "configs")
DEFAULT_SOURCE_CONFIG = "dreyer_basenet.yaml"
DEFAULT_CONFIG = "dreyer_finetune.yaml"


def finetune(source_config: dict, config: dict):
    run_dir, ckpt_name, n_subjects = config.get("run_dir"), config.get("ckpt_name"), config.get("n_subjects")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cls = get_model_cls(source_config.get("model"))
    source_dm = source_config.get("datamodule", "LargeDatasetDataModuleLMSO")
    datamodule_cls = get_datamodule_cls(source_dm[:-4] + "Within")

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

        if config.get("reset_bn", False):
            model = reset_bn(model, datamodule.train_dataset)

        finetuner = ClassificationModule(model.model, **config.get("finetuner_kwargs"),
                                         max_epochs=config.get("max_epochs"))
        trainer = Trainer(
            max_epochs=config.get("max_epochs"),
            num_sanity_val_steps=0,
            accelerator="auto",
            strategy="auto",
            enable_checkpointing=False
        )
        trainer.fit(finetuner, datamodule)

        # run-wise results
        # prediction probability per window (n_trials x n_windows)
        y_pred = torch.concat(
            trainer.predict(model, datamodule.predict_dataloader()), dim=-1).T

        # overall accuracies (trial-wise)
        y_test = datamodule.test_dataset.tensors[1]
        taccs = ((y_pred.mean(dim=-1) > 0.5).float() == y_test).float()

        # write to dataframe
        trials_per_run = [len(datamodule.dataset.data_dict[subject_id]["labels"]
                              [f"run_{i}"]) for i in datamodule.dataset.test_run_ids]
        trials_counter = 0
        for run_idx, run_id in enumerate(results_df.columns):
            results_df.at[subject_id, run_id] = taccs[trials_counter:trials_counter + trials_per_run[run_idx]].mean().item()
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

    finetune(source_config, config)
