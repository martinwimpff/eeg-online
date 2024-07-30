from torch.utils.data import Dataset, TensorDataset


class WindowDataSet(Dataset):
    """
    This is a wrapper class for TensorDatasets.
    It ensures easy window loading while remaining memory-efficient
    """
    def __init__(self, dataset: TensorDataset, n_windows_per_trial: int = 61):
        self.dataset = dataset
        self.n_windows_per_trial = n_windows_per_trial

    def __len__(self):
        return self.dataset.tensors[-1].shape[0] * self.n_windows_per_trial

    def __getitem__(self, idx):
        trial_idx = idx // self.n_windows_per_trial
        window_idx = idx % self.n_windows_per_trial
        window = self.dataset.tensors[0][trial_idx, :, window_idx*16:window_idx*16+256]
        y = self.dataset.tensors[1][trial_idx]
        return window, y