# eeg-online
## Usage
### Data
Download the datasets from 
[zenodo(Dreyer)](https://zenodo.org/records/8089820) and 
[GigaDB(Lee)](http://gigadb.org/dataset/100542).
Place the data in the [data](data) folder.
### Installation
Run ``pip install .`` to install the ``eeg-online`` package.

_Note: you can also use poetry for the installation_
### Source training
Run [train_test.py](eeg_online/train_test.py) with a ``--config`` of your choice.
### Supervised and unsupervised finetuning
Run [finetune.py](eeg_online/finetune.py) with the corresponding 
``--source_config`` and a ``--config`` of your choice.

We included some checkpoints (for the first 5 subjects) 
in [lightning_logs](eeg_online/lightning_logs) to provide a starting point.

### Online test-time adaptation
Run [run_online_adaptation.py](eeg_online/tta/run_online_adaptation.py) with the corresponding 
``--source_config`` and a ``--config`` of your choice.

In case of any specific questions please
[contact me](mailto:martin.wimpff@iss.uni-stuttgart.de) or create an issue.

## Citation
If you find this repository useful, please cite our work
```
@article{202407.2370,
	doi = {10.20944/preprints202407.2370.v1},
	url = {https://doi.org/10.20944/preprints202407.2370.v1},
	year = 2024,
	month = {July},
	publisher = {Preprints},
	author = {Martin Wimpff and Jan Zerfowski and Bin Yang},
	title = {Tailoring Deep Learning for Real-Time Brain-Computer Interfaces: From Offline Models to Calibration-Free Online Decoding},
	journal = {Preprints}
}
```
