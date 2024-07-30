import os
from pathlib import Path

LIGHTNING_LOGS = os.path.join(Path(__file__).resolve().parents[1], "lightning_logs")


def load_checkpoint(run_dir, version, ckpt_name):
    ckpt_path = os.path.join(LIGHTNING_LOGS, run_dir, version, "checkpoints", ckpt_name)

    if os.path.isfile(ckpt_path):
        return ckpt_path
    else:
        raise FileNotFoundError
