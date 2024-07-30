import os
from pathlib import Path

import mne
import numpy as np

DATA_PATH = os.path.join(Path(__file__).resolve().parents[2], "data", "BCI Database",
                         "BCI Database", "Signals")


def load_run(subject_id: int, run_id: int, l_freq: int = 5, h_freq: int = 35,
             baseline_correction: bool = False, sfreq: int = 256):
    if subject_id <= 60:
        dataset = "A"
    elif (subject_id > 60) and (subject_id <= 81):
        dataset = "B"
    else:
        dataset = "C"
    run_type = "acquisition" if run_id <= 2 else "onlineT"

    # load raw
    subject_path = os.path.join(DATA_PATH, f"DATA {dataset}", f"{dataset}{subject_id}")
    raw = mne.io.read_raw_gdf(os.path.join(
        subject_path, f"{dataset}{subject_id}_R{run_id}_{run_type}.gdf"), preload=True)
    ch_names = raw.info["ch_names"]
    selected_channels = [ch_name for ch_name in ch_names if not(
            ("EOG" in ch_name) or ("EMG" in ch_name))]
    raw = raw.pick_channels(selected_channels)
    raw.info.set_montage(mne.channels.make_standard_montage("standard_1020"))

    # filter data
    if (l_freq is not None) or (h_freq is not None):
        raw = raw.filter(l_freq, h_freq)

    # create epochs and labels
    events, event_ids = mne.events_from_annotations(raw)
    selected_event_ids = {key: event_ids.get(key) for key in ["769", "770"]}
    baseline = (-3.0, -1.0) if baseline_correction else None
    epochs = mne.Epochs(raw, events, selected_event_ids, tmin=-3.0, tmax=5.0,
                        baseline=baseline, preload=True)
    epochs.resample(sfreq)
    y = np.where(epochs.events[:, -1] <= event_ids["769"], 0, 1)
    # 769 -> 5 -> 0, LEFT HAND
    # 770 -> 6 -> 1, RIGHT HAND

    return epochs, y
