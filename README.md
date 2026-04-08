The code implementation of our paper "Spectral Dynamic Attention Network for Hyperspectral Image Super-Resolution", submitted to IEEE GRSL.

## Requirements
* basicsr 1.3.4
* Python 3.8
* PyTorch  1.11.0
* CUDA  11.3

## Preparation
To get the training set, validation set and testing set, refer to SSPSR to download the mcodes for cropping the hyperspectral image.

## Training
To train SDANet, run the following command from the project root:<br>
```
./run_train.sh 0
```
Use another GPU by changing the index, e.g. `./run_train.sh 2`.

## Testing
To test SDANet, run the following command from the project root:<br>
```
./run_test.sh 0
```
Use another GPU by changing the index, e.g. `./run_test.sh 2`.
## References
* [SSPSR](https://github.com/junjun-jiang/SSPSR)
