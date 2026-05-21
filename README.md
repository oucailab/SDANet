# 🚀 Spectral Dynamic Attention Network for Hyperspectral Image Super-Resolution, IEEE GRSL 2026.

<p align="center">
  <a href="https://ieeexplore.ieee.org/document/11505914">
    <img src="https://img.shields.io/badge/Paper-IEEE%20GRSL%20-1f6feb?style=for-the-badge" alt="Paper Button" />
  </a>
  <a href="https://arxiv.org/abs/2604.27326">
    <img src="https://img.shields.io/badge/Paper-ARXIV-brightgreen?style=for-the-badge" alt="Paper Button" />
  </a>
</p>

## 📌 **Introduction**

This repository contains the official implementation of our paper:  
📄 *Spectral Dynamic Attention Network for Hyperspectral Image Super-Resolution* *(IEEE GRSL 2026)*  

## 🛠 Requirements
* basicsr 1.3.4
* Python 3.8
* PyTorch  1.11.0.
* CUDA  11.3

## 🏋️‍♂️ Preparation
To get the training set, validation set and testing set, refer to SSPSR to download the mcodes for cropping the hyperspectral image.

## Training
To train SDANet, run the following command from the project root:<br>
```
./run_train.sh 0
```
Use another GPU by changing the index, e.g. `./run_train.sh 2`.

You can also pass the option file directly (same style as manual command):
```
./run_train.sh options/train/HSI/SDAnetChux4.yml 0
```

Equivalent direct command:
```
CUDA_VISIBLE_DEVICES=0 python ./basicsr/train.py -opt ./options/train/HSI/SDAnetChux4.yml
```

## Testing
To test SDANet, run the following command from the project root:<br>
```
./run_test.sh 0
```
Use another GPU by changing the index, e.g. `./run_test.sh 2`.

You can also pass the option file directly:
```
./run_test.sh options/test/HSI/test_SDAnetChux4.yml 0
```

Equivalent direct command:
```
CUDA_VISIBLE_DEVICES=0 python ./basicsr/test.py -opt ./options/test/HSI/test_SDAnetChux4.yml
```

