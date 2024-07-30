import os
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat

DATA_PATH = os.path.join(Path(__file__).resolve().parents[2], "data",
                         "MNE-lee2019-mi-data")


def load_run(subject_id: int, run_id: int, l_freq: int = 8, h_freq: int = 30,
             baseline_correction: bool = False, sfreq: int = 256):
    path = os.path.join(DATA_PATH, f"session{run_id}", f"s{subject_id}",
                        f"sess0{run_id}_subj{subject_id:02d}_EEG_MI.mat")
    mat = loadmat(path)
    data = mat["EEG_MI_train"][0, 0]

    fs = data["fs"].item()
    ch_names = [np.squeeze(c).item() for c in np.ravel(data["chan"])]
    info = mne.create_info(ch_names, ch_types=["eeg"] * len(ch_names), sfreq=fs)
    raw = mne.io.RawArray(data["x"].T*1e-6, info=info)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
    raw.pick_channels([
        "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"
    ])

    # filter data
    if (l_freq is not None) or (h_freq is not None):
        raw = raw.filter(l_freq, h_freq)

    event_times_in_samples = data["t"].squeeze()
    labels = data["y_dec"].squeeze()
    events = np.zeros((len(event_times_in_samples), 3), dtype="int")
    events[:, 0] = event_times_in_samples
    events[:, -1] = labels
    event_id = {"1": 1, "2": 2}

    baseline = (-3.0, -1.0) if baseline_correction else None
    epochs = mne.Epochs(raw, events, event_id, tmin=-3.0, tmax=4.0,
                        baseline=baseline, preload=True)
    epochs.resample(sfreq)

    y = np.where(labels >= 2, 0, 1)  # left (2) -> 0, right (1) -> 1

    return epochs, y
