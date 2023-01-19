# Synthetic Battery Data
Generation of synthetic battery operation data using a generative adversarial network (GAN) and a state-of-charge (SOC) estimator.

<!-- ![Generative framework overview](./generative_framework.png) -->
<p align="center">
  <img src="./generative_framework.png" width="550">
</p>

## Introduction

This repository provides the implementation of a fusion approach combining a GAN and an [**SOC estimator**](https://github.com/KeiLongW/battery-state-estimation) for synthetic battery operation data generation.
The experiment is performed on the public battery dataset: [**LG 18650HG2 Li-ion Battery Data**](https://data.mendeley.com/datasets/cp3473x7xv/3).

## Publication

The work shown here is an implementation from a research article that is currently being reviewed:

*A Novel Fusion Approach Consists of GAN and State-of-Charge Estimator for Synthetic Battery Operation Data Generation*

## Project explanation

- `data` directory: contains the preprocessed data `lg_600_data.npy` of [**LG 18650HG2 Li-ion Battery Data**](https://data.mendeley.com/datasets/cp3473x7xv/3) that is reshape to $[n,600,4]$. Please refer to [**SOC estimator**](https://github.com/KeiLongW/battery-state-estimation) for detail handling of the preprocessing. 
- `result` directory: the training result will be output to this directory
- `soc_models` directory: contains the pre-trained model of the SOC estimator used in this work (download from [here](https://github.com/KeiLongW/battery-state-estimation/releases/tag/v1.0)).
- `lg_dataset.py`: code to handle the LG dataset.
- `gan_trainer_base.py`: code the handle the GAN training procedure.
- `train_lstm_gan.ipynb`: jupyter notebook for evaluating the training result.
- `train_lstm_gan.py`: main code for the proposed GAN model.

## Quick start

### Install requirements:
Create conda environment from our yml file. The version of pytorch and tensorflow may differ depends on your server's configuration.
```
conda env create -f environment.yml
```

### Available arguments
Use below command to view the available arguments. The default arguments are the parameters used in our paper.
```
python3 train_lstm_gan.py -h
```

### Training
Start training with the default parameters.
```
python3 train_lstm_gan.py
```

### Evaluation
Once you start the training, a new result will be created in the `results` directory. Run the `result.ipynb` notebook to view the training result.
Alternatively, you can download our result form release and place to the `results` directory to evaluate our trained model.