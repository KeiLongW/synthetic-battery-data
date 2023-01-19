import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from arg_parser import parse_arg
from gan_trainer_base import GANTrainerBase
from lg_dataset import LGDataset


############################################
# Models definition
############################################
class Generator(nn.Module):
  def __init__(self, input_dim, output_dim, seq_len, dropout):
    super(Generator, self).__init__()
    
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.seq_len = seq_len
    self.dropout = dropout
    
    self.lstm_1 = nn.LSTM(input_dim, 1024, dropout=dropout, batch_first=True)
    self.batch_norm_1 = nn.BatchNorm2d(1024)
    self.relu_1 = nn.ReLU()
    self.lstm_2 = nn.LSTM(1024, 512, dropout=dropout, batch_first=True)
    self.batch_norm_2 = nn.BatchNorm2d(512)
    self.relu_2 = nn.ReLU()
    self.lstm_3 = nn.LSTM(512, 256, dropout=dropout, batch_first=True)
    self.batch_norm_3 = nn.BatchNorm2d(256)
    self.relu_3 = nn.ReLU()
    self.lstm_4 = nn.LSTM(256, 128, dropout=dropout, batch_first=True)
    self.batch_norm_4 = nn.BatchNorm2d(128)
    self.relu_4 = nn.ReLU()
    self.lstm_5 = nn.LSTM(128, output_dim, dropout=dropout, batch_first=True)
    
  def forward(self, x):
    self.lstm_1.flatten_parameters()
    self.lstm_2.flatten_parameters()
    self.lstm_3.flatten_parameters()
    self.lstm_4.flatten_parameters()
    self.lstm_5.flatten_parameters()
    
    x, (h_n, c_n) = self.lstm_1(x)
    x = self.batch_norm_1(x.reshape(x.shape[0], x.shape[2], 1, x.shape[1])).squeeze(2).permute(0,2,1)
    x = self.relu_1(x)
    x, (h_n, c_n) = self.lstm_2(x)
    x = self.batch_norm_2(x.reshape(x.shape[0], x.shape[2], 1, x.shape[1])).squeeze(2).permute(0,2,1)
    x = self.relu_2(x)
    x, (h_n, c_n) = self.lstm_3(x)
    x = self.batch_norm_3(x.reshape(x.shape[0], x.shape[2], 1, x.shape[1])).squeeze(2).permute(0,2,1)
    x = self.relu_3(x)
    x, (h_n, c_n) = self.lstm_4(x)
    x = self.batch_norm_4(x.reshape(x.shape[0], x.shape[2], 1, x.shape[1])).squeeze(2).permute(0,2,1)
    x = self.relu_4(x)
    x, (h_n, c_n) = self.lstm_5(x)
    return x
  
class Discriminator(nn.Module):
  def __init__(self, input_dim, dropout):
    super(Discriminator, self).__init__()
    
    self.input_dim = input_dim
    self.dropout = dropout
    
    self.lstm_1 = nn.LSTM(input_dim, 128, dropout=dropout, batch_first=True)
    self.instance_norm_1 = nn.InstanceNorm2d(128)
    self.relu_1 = nn.LeakyReLU(0.2)
    self.lstm_2 = nn.LSTM(128, 256, dropout=dropout, batch_first=True)
    self.instance_norm_2 = nn.InstanceNorm2d(256)
    self.relu_2 = nn.LeakyReLU(0.2)
    self.lstm_3 = nn.LSTM(256, 512, dropout=dropout, batch_first=True)
    self.instance_norm_3 = nn.InstanceNorm2d(512)
    self.relu_3 = nn.LeakyReLU(0.2)
    self.lstm_4 = nn.LSTM(512, 1024, dropout=dropout, batch_first=True)
    self.instance_norm_4 = nn.InstanceNorm2d(1024)
    self.relu_4 = nn.LeakyReLU(0.2)
    self.linear = nn.Linear(1024, 1)

  def forward(self, x):
    self.lstm_1.flatten_parameters()
    self.lstm_2.flatten_parameters()
    self.lstm_3.flatten_parameters()
    self.lstm_4.flatten_parameters()
    
    x, (h_n, c_n) = self.lstm_1(x)
    x = self.instance_norm_1(x.reshape(x.shape[0], x.shape[2], 1, x.shape[1])).squeeze(2).permute(0,2,1)
    x = self.relu_1(x)
    x, (h_n, c_n) = self.lstm_2(x)
    x = self.instance_norm_2(x.reshape(x.shape[0], x.shape[2], 1, x.shape[1])).squeeze(2).permute(0,2,1)
    x = self.relu_2(x)
    x, (h_n, c_n) = self.lstm_3(x)
    x = self.instance_norm_3(x.reshape(x.shape[0], x.shape[2], 1, x.shape[1])).squeeze(2).permute(0,2,1)
    x = self.relu_3(x)
    x, (h_n, c_n) = self.lstm_4(x)
    x = self.instance_norm_4(x.reshape(x.shape[0], x.shape[2], 1, x.shape[1])).squeeze(2).permute(0,2,1)
    output = self.relu_4(x[:,-1,:])
    output = self.linear(output)
    return output
  
  
############################################
# Trainer
############################################
class TrainLSTMGAN(GANTrainerBase):
  def __init__(self, config) -> None:
    super().__init__(config, 'lstm_gan')
    
    ########## Config ##########
    self.generator_dropout = config.get('generator_dropout')
    self.generator_learning_rate = config.get('generator_learning_rate')
    self.generator_weight_decay = config.get('generator_weight_decay')
    self.generator_adam_beta1 = config.get('generator_adam_beta1')
    self.generator_adam_beta2 = config.get('generator_adam_beta2')
    self.discriminator_dropout = config.get('discriminator_dropout')
    self.discriminator_learning_rate = config.get('discriminator_learning_rate')
    self.discriminator_weight_decay = config.get('discriminator_weight_decay')
    self.discriminator_adam_beta1 = config.get('discriminator_adam_beta1')
    self.discriminator_adam_beta2 = config.get('discriminator_adam_beta2')
    
    ########## GAN ##########
    self.G = Generator(self.generator_input_dim, 
                       self.generator_output_dim,
                       self.train_set.seq_len,
                       self.generator_dropout)
    self.G = DataParallel(self.G, device_ids=self.device_ids).to(self.device)
    self.g_optim = Adam(self.G.parameters(), lr=self.generator_learning_rate, weight_decay=self.generator_weight_decay, betas=(self.generator_adam_beta1, self.generator_adam_beta2))
    self.D = Discriminator(self.discriminator_input_dim,
                           self.discriminator_dropout)
    self.D = DataParallel(self.D, device_ids=self.device_ids).to(self.device)
    self.d_optim = Adam(self.D.parameters(), lr=self.discriminator_learning_rate, weight_decay=self.discriminator_weight_decay, betas=(self.discriminator_adam_beta1, self.discriminator_adam_beta2))
    
def main():
  args = parse_arg()
  
  feature_col_index = [LGDataset.Column.voltage, LGDataset.Column.current, LGDataset.Column.temperature]
  trainer = TrainLSTMGAN({
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'device_ids': list(range(torch.cuda.device_count())),
    'train_report_interval': args.train_report_interval,
    'batch_size': args.batch_size,
    'data_path': args.data_path,
    'feature_col_index': feature_col_index,
    'show_progress_bar': args.show_progress_bar,
    'resume_model_name': args.resume_model_name,
    'save_epoch_interval': args.save_epoch_interval,
    'result_notebook': args.result_notebook,
    
    'gan_epochs': args.gan_epochs,
    'generator_input_dim': args.generator_input_dim,
    'generator_output_dim': len(feature_col_index),
    'discriminator_train_steps': args.discriminator_train_steps,
    'grad_penalty_lambda_term': args.gradient_penalty_lambda_term,
    
    'generator_dropout': args.generator_dropout,
    'generator_learning_rate': args.generator_learning_rate,
    'generator_weight_decay': args.generator_weight_decay,
    'generator_adam_beta1': args.generator_adam_beta1,
    'generator_adam_beta2': args.generator_adam_beta2,
    'discriminator_dropout': args.discriminator_dropout,
    'discriminator_learning_rate': args.discriminator_learning_rate,
    'discriminator_weight_decay': args.discriminator_weight_decay,
    'discriminator_adam_beta1': args.discriminator_adam_beta1,
    'discriminator_adam_beta2': args.discriminator_adam_beta2,
    
    'evaluate_sample_size': args.evaluate_sample_size,
    'soc_estimator_step': args.soc_estimator_step,
    'soc_estimator_model_path': args.soc_estimator_model_path,
    'soc_estimator_model': args.soc_estimator_model,
  })
  
  trainer.copy_own_py_file(__file__)
  trainer.train()
  
if __name__ == '__main__':
  main()