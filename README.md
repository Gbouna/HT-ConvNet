# HT-ConvNet
This is the official repository for **Hierarchical Temporal Convolution Network: Towards Privacy-Centric Activity Recognition**, our paper published at the International Conference on Ubiquitous Computing and Ambient Intelligence (UCAmI)

# Prerequisites
Python >= 3.6

PyTorch >= 1.1.0

PyYAML, tqdm
`
# Data Preparation

## Download datasets.
Download the JHMDB and SHREC datasets using the links below:
```
JHMDB raw data download link:   http://jhmdb.is.tue.mpg.de/challenge/JHMDB/datasets
SHREC raw data download link:   http://www-rech.telecom-lille.fr/shrec2017-hand/
```

## Process datasets.
Use the preprocessing code in the data processing folder to process the data and put them in the data folder. 

Or you can get the already processed data directly from [this GitHub repo](https://github.com/fandulu/DD-Net)

A sample of the processed file is currently in the data folder, please replace it. 

# Training

For JHMDB, run `python train.py --batch-size 512 --epochs 600 --dataset 0 --lr 0.001 | tee train.log`

For SHREC coarse, run `python train.py --batch-size 512 --epochs 600 --dataset 1 --lr 0.001 | tee train.log`

For SHREC fine, run `python train.py --batch-size 512 --epochs 600 dataset 2 --lr 0.001 | tee train.log`

# Testing

To test the trained model, bring the saved model to the main directory and pass its name as an arg for the model-path or simply pass the path to where the model was saved

For JHMDB, run `python test.py --model-path model.pt --dataset 0`

For SHREC coarse, run `python test.py --model-path model.pt --dataset 1`

For SHREC fine, run `python test.py --model-path model.pt --dataset 2`

To force the model to be loaded with CPU run `python test.py --model-path model.pt --dataset 0 --no-cuda`

# Action Recognition in Real-time

![Real-time Action Recognition](https://github.com/user-attachments/assets/cf372538-8c3a-45e5-a46a-87150a166bd6)




