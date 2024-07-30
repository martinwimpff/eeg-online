from argparse import ArgumentParser
import os
from pathlib import Path
import yaml

import pandas as pd
from pytorch_lightning import Trainer
import torch

from eeg_online.utils.get_datamodule_cls import get_datamodule_cls
from eeg_online.utils.get_model_cls import get_model_cls
from eeg_online.utils.seed import seed_everything

CONFIG_DIR = os.path.join(Path(__file__).resolve().parents[1], "configs")
DEFAULT_CONFIG = "lee_basenet.yaml"


def train_and_test(config: dict):
    model_cls = get_model_cls(model_name=config["model"])
    datamodule_cls = get_datamodule_cls(config["datamodule"])
    datamodule = datamodule_cls(
        n_subjects=config.get("n_subjects"), n_folds=config.get("n_folds"),
        preprocessing_dict=config.get("preprocessing").copy()
    )
    results_df = pd.DataFrame(index=datamodule.dataset.all_subject_ids,
                              columns=[f"run_{i}" for i in datamodule.dataset.test_run_ids])

    for fold_idx in range(config.get("n_folds")):
        seed_everything(config.get("seed"))
        datamodule.setup_fold(fold_idx)

        trainer = Trainer(
            max_epochs=config.get("max_epochs"),
            num_sanity_val_steps=0,
            accelerator="auto",
            strategy="auto",
            enable_checkpointing=config.get("log_model", False),
            logger=None
        )
        model = model_cls(**config.get("model_kwargs"),
                          max_epochs=config.get("max_epochs"))
        trainer.fit(model, datamodule=datamodule)

        # run-wise results
        # prediction probability per window (n_trials x n_windows)
        y_pred = torch.concat(
            trainer.predict(model, datamodule.predict_dataloader()), dim=-1).T

        # overall accuracies (trial-wise)
        y_test = datamodule.test_dataset.tensors[1]
        taccs = ((y_pred.mean(dim=-1) > 0.5).float() == y_test).float()

        # write to dataframe
        test_subject_ids = datamodule.get_test_subject_ids(fold_idx)
        trials_counter = 0
        trials_per_run = {
            subject_id: [
                len(datamodule.dataset.data_dict[subject_id]["labels"][f"run_{i}"])
                for i in datamodule.dataset.test_run_ids]
            for subject_id in test_subject_ids}
        for subject_id in test_subject_ids:
            for run_idx, run in enumerate([f"run_{i}" for i in datamodule.dataset.test_run_ids]):
                results_df.at[subject_id, run] = taccs[trials_counter:trials_counter + trials_per_run[subject_id][run_idx]].mean().item()
                trials_counter += trials_per_run[subject_id][run_idx]

    print("results per subject")
    print(results_df.mean(1))
    print("results per run")
    print(results_df.mean(0))


if __name__ == "__main__":
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    # load config
    with open(os.path.join(CONFIG_DIR, args.config)) as f:
        config = yaml.safe_load(f)

    train_and_test(config)
