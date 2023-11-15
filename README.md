# SCONE

- This GitHub repo is the implementation of "SCONE: A Food Scooping Robot Learning Framework with Active Perception" in CoRL 2023.
[[openreview](https://openreview.net/forum?id=yHlUVHWnBN)] [[website](https://sites.google.com/view/corlscone)]

<p align="center">
  <img width="600" height="300" src="images/sample.gif">
</p>

## System Requirements
- Linux (Teseted on Ubuntu 18.04)
- Python 3 (Tested on Python 3.7)
- Torch (Tested on Torch 1.9.1)
- Cuda (Tested on Cuda 11.4)
- GPU (Tested on Nvidia RTX3090)
- CPU (Tested on Intel COre i7-10700)

## Setup
- Clone This Repo
```
$ git clone https://github.com/HCIS-Lab/SCONE.git
```
- Create Conda Environment
```
$ cd SCONE
$ conda create -n scone python=3.7
$ conda activate scone
$ pip install -r requirements.txt
```

## Usage
```
python3 train.py --dataset <data_root>
```

## TODO
- Upload raw & processed dataset
- complete the Class FoodDataset in load_data.py

## Citation
```
@inproceedings{tai2023scone,
  title={SCONE: A Food Scooping Robot Learning Framework with Active Perception},
  author={Tai, Yen-Ling and Chiu, Yu Chien and Chao, Yu-Wei and Chen, Yi-Ting},
  booktitle={7th Annual Conference on Robot Learning},
  year={2023}
}
```
