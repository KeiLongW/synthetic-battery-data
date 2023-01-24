import json
import math
import os
import shutil
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow import keras
from torch import autograd
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from lg_dataset import LGDataset


class GANTrainerBase(ABC):
  def __init__(self, config, trainer_name) -> None:
    self.trainer_name = trainer_name
    self.result_base_dir = './results/'
    self.result_name = str(int(time.time())) + '_' + self.trainer_name
    
    ########## Config ##########
    self.config = config
    self.device = torch.device(config.get('device'))
    self.device_ids = config.get('device_ids')
    self.train_report_interval = config.get('train_report_interval')
    self.batch_size = config.get('batch_size')
    self.data_path = config.get('data_path')
    self.feature_col_index = config.get('feature_col_index')
    self.show_progress_bar = config.get('show_progress_bar')
    self.resume_model_name = config.get('resume_model_name')
    self.save_epoch_interval = config.get('save_epoch_interval')
    self.result_notebook = config.get('result_notebook')
    
    self.gan_epochs = config.get('gan_epochs')
    self.generator_input_dim = config.get('generator_input_dim')
    self.generator_output_dim = config.get('generator_output_dim')
    self.discriminator_input_dim = self.generator_output_dim
    self.discriminator_train_steps = config.get('discriminator_train_steps')
    self.grad_penalty_lambda_term = config.get('grad_penalty_lambda_term')
    
    self.evaluate_sample_size = config.get('evaluate_sample_size')
    self.soc_estimator_step = config.get('soc_estimator_step')
    self.soc_estimator_model_path = config.get('soc_estimator_model_path')
    self.soc_estimator_model = config.get('soc_estimator_model')
    
    ########## Data ##########
    self.train_set = LGDataset(self.data_path, self.feature_col_index)
    self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
    self.real_set = LGDataset(self.data_path)
    
    ########## SOC Estimator ##########
    physical_devices = tf.config.list_physical_devices('GPU')
    # avoid the keras model allocate all the gpu memory
    for device in physical_devices:
      tf.config.experimental.set_memory_growth(device, True)
    self.soc_model = keras.models.load_model(self.soc_estimator_model_path + self.soc_estimator_model)
    
  def _save_model(self, model, model_name):
    model_path = f'{self.result_base_dir}{self.result_name}/{model_name}.pth'
    torch.save(model.state_dict(), model_path)
    
  def _save_json(self, json_content, name):
    with open(f'{self.result_base_dir}{self.result_name}/{name}.json', 'w') as f:
      json.dump(json_content, f, indent=2)
    
  def _create_result_dir(self):
    path = f'{self.result_base_dir}{self.result_name}'
    if not os.path.exists(path):
      os.makedirs(path)
      
  def _do_training(self):
    begin_time = datetime.now()    
    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1    
    
    for epoch in range(self.start_epoch, self.gan_epochs):    
      self.D.train()
      self.G.train()      
      ########## Train discriminator ##########
      for p in self.D.parameters():
        p.requires_grad = True
  
      d_loss_real = 0
      d_loss_fake = 0
      total_d_loss = 0
      d_iterator = tqdm(range(self.discriminator_train_steps), desc=f'Epoch {epoch+1}/{self.gan_epochs}') if self.show_progress_bar else range(self.discriminator_train_steps)
      for d_step in d_iterator:
        self.d_optim.zero_grad()
        real_data = next(iter(self.train_loader)).to(self.device) # randomly get one batch of real data
        d_loss_real = self.D(real_data)
        d_loss_real = d_loss_real.mean()
        d_loss_real.backward(mone)
        
        z = torch.randn(self.batch_size, self.train_set.seq_len, self.generator_input_dim).to(self.device)
        fake_data = self.G(z)
        d_loss_fake = self.D(fake_data)
        d_loss_fake = d_loss_fake.mean()
        d_loss_fake.backward(one)
        
        with torch.backends.cudnn.flags(enabled=False):
          gradient_penalty = self._calculate_gradient_penalty(real_data.data, fake_data.data)
          gradient_penalty.backward()
        
        self.d_optim.step()
        total_d_loss += (d_loss_fake - d_loss_real + gradient_penalty).item()
        print(f'[D iter:{d_step+1}/{self.discriminator_train_steps}] d_loss_real:{d_loss_real}, d_loss_fake:{d_loss_fake}, gradient_penalty:{gradient_penalty}')
      d_loss = total_d_loss / self.discriminator_train_steps
        
      ########## Train generator ##########
      for p in self.D.parameters():
        p.requires_grad = False
        
      self.g_optim.zero_grad()
      z = torch.randn(self.batch_size, self.train_set.seq_len, self.generator_input_dim).to(self.device)
      fake_data = self.G(z)
      g_loss = self.D(fake_data)
      g_loss = g_loss.mean()
      g_loss.backward(mone)
      self.g_optim.step()
      g_loss = -g_loss.item() # reverse the sign of g_loss to have a more intuitive interpretation of loss
      
      self.g_loss_list.append(g_loss)
      self.d_loss_list.append(d_loss)   
      if (epoch + 1) % self.train_report_interval == 0:
        print(f'[Epoch:{epoch+1}/{self.gan_epochs}] Time passed:{(datetime.now() - begin_time).seconds}s, g_loss:{g_loss}, d_loss:{d_loss}')
    
      self._training_snapshot(epoch, g_loss, d_loss, begin_time)
        
  def _calculate_gradient_penalty(self, real_data, fake_data):
    eta = torch.FloatTensor(self.batch_size,1,1).uniform_(0,1)
    eta = eta.expand(self.batch_size, real_data.size(1), real_data.size(2)).to(self.device)
    
    interpolated = eta * real_data + ((1 - eta) * fake_data)
    interpolated = interpolated.to(self.device)
    
    interpolated = Variable(interpolated, requires_grad=True)
    
    prob_interpolated = self.D(interpolated)
    
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                              create_graph=True, retain_graph=True)[0]
    
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.grad_penalty_lambda_term
    return grad_penalty
  
  def _training_snapshot(self, epoch, g_loss, d_loss, begin_time):
    self._save_model(self.G, 'generator')
    self._save_model(self.D, 'discriminator')
    self._save_json(self.g_loss_list, 'g_loss')
    self._save_json(self.d_loss_list, 'd_loss')
    if g_loss < self.best_g_loss:
      self.best_g_loss = g_loss
      self._save_model(self.G, 'best_generator')
      self._save_json( {'loss': self.best_g_loss, 'epoch': epoch+1}, 'best_generator')
    if d_loss < self.best_d_loss:
      self.best_d_loss = d_loss
      self._save_model(self.D, 'best_discriminator')
      self._save_json({'loss': self.best_d_loss, 'epoch': epoch+1}, 'best_discriminator')
    if self.save_epoch_interval is not None and (epoch+1) % self.save_epoch_interval == 0:
      self._do_evaluation(epoch, begin_time)
  
  def _do_evaluation(self, epoch, begin_time):
    print('Start evaluation...')
    epoch_dir = f'{epoch+1}_epoch/'
    epoch_full_dir = f'{self.result_base_dir}{self.result_name}/{epoch_dir}'
    os.makedirs(epoch_full_dir)
    self._save_model(self.G, f'{epoch_dir}generator')
    self._save_model(self.D, f'{epoch_dir}discriminator')
    
    ########## Save fake data sample ##########
    fake_data = np.empty((0, self.train_set.seq_len, len(self.feature_col_index)))
    for _ in range(math.ceil(self.evaluate_sample_size / self.batch_size)):
      result = self.generate(self.batch_size, self.train_set.seq_len)
      fake_data = np.concatenate((fake_data, result), axis=0)
    fake_data_soc_est_shape, sample_len = self._sliding_window(fake_data)
    fake_data_soc = self.soc_model.predict(fake_data_soc_est_shape, verbose=0)
    fake_data = np.concatenate((fake_data[:,fake_data.shape[1]-sample_len-1:fake_data.shape[1]-1,:], 
                                fake_data_soc.reshape((fake_data.shape[0], sample_len, 1))), axis=-1)
    np.save(epoch_full_dir+'fake_data.npy', fake_data)
    
    ########## Save data distribution ##########
    real_idx = np.random.randint(len(self.real_set.data), size=self.evaluate_sample_size)
    real_data = self.real_set.data[real_idx].numpy()
    real_data = real_data[:,:sample_len,:]
    fake_data_dist = np.empty((0, sample_len))
    real_data_dist = np.empty((0, sample_len))
    for i in range(self.evaluate_sample_size):
      fake_data_dist = np.concatenate((fake_data_dist, 
                                  np.reshape(np.mean(fake_data[i,:,:],1), [1,sample_len])))
      real_data_dist = np.concatenate((real_data_dist, 
                                  np.reshape(np.mean(real_data[i,:,:],1), [1,sample_len])))
    
    ########## PCA ##########
    pca = PCA(n_components=2)
    pca.fit(real_data_dist)
    pca_real_features = pca.transform(real_data_dist)
    pca_fake_features = pca.transform(fake_data_dist)
    np.save(epoch_full_dir+'pca_real.npy', pca_real_features)
    np.save(epoch_full_dir+'pca_fake.npy', pca_fake_features)
    
    ########## t-SNE ##########
    tsne = TSNE(n_components = 2, init='pca', learning_rate='auto')
    tsne_features = tsne.fit_transform(np.concatenate((real_data_dist, fake_data_dist), axis = 0))
    np.save(epoch_full_dir+'tsne_real.npy', tsne_features[:real_data_dist.shape[0]])
    np.save(epoch_full_dir+'tsne_fake.npy', tsne_features[real_data_dist.shape[0]:])
    
    print(f'Finished evaluation, time passed:{(datetime.now() - begin_time).seconds}s')
    
  def _sliding_window(self, data):
    sample_len = min(self.soc_estimator_step, data.shape[1]-self.soc_estimator_step)
    data_reshape = []
    for sample in data:
      for i in range(sample_len):
        data_reshape.append(sample[i:i+self.soc_estimator_step])
    return np.array(data_reshape), sample_len
        
  def _init_training(self):
    ########## Training initialization ##########    
    self.best_g_loss = float('inf')
    self.best_d_loss = float('inf')
    self.g_loss_list = []
    self.d_loss_list = []
    self.start_epoch = 0
    if self.resume_model_name is not None:
      model_path = self.result_base_dir + self.resume_model_name + '/{model_name}.pth'
      self.load_model(model_path)
      self._load_loss(self.resume_model_name)
      self.start_epoch = len(self.g_loss_list)
        
  def train(self):            
    self._create_result_dir()
    self._init_training()
    
    print(self.config)
    print('Generator:')
    summary(self.G, (1, self.train_set.seq_len, self.generator_input_dim))
    print('Discriminator:')
    summary(self.D, (1, self.train_set.seq_len, self.generator_output_dim))
    
    self._save_json(self.config, 'config')
    self.copy_own_py_file(__file__)
    shutil.copyfile('lg_dataset.py', f'{self.result_base_dir}{self.result_name}/lg_dataset.py')
    shutil.copyfile('arg_parser.py', f'{self.result_base_dir}{self.result_name}/arg_parser.py')
    if self.result_notebook is not None:
      shutil.copyfile(self.result_notebook, f'{self.result_base_dir}{self.result_name}/result.ipynb')
    
    print('Start training: ', self.result_name)
    print(f'Training set: {self.train_set.data_shape}')
    
    self._do_training()
      
  def copy_own_py_file(self, file):
    self._create_result_dir()
    shutil.copy(file, f'{self.result_base_dir}{self.result_name}/{os.path.basename(file)}')
    
  def load_model(self, model_path):
    self.G.load_state_dict(torch.load(model_path.format(model_name='generator'), map_location=self.config.get('device')))
    self.D.load_state_dict(torch.load(model_path.format(model_name='discriminator'), map_location=self.config.get('device')))
    
  def _load_loss(self, result_name):
    g_loss_path = self.result_base_dir + result_name + '/g_loss.json'
    d_loss_path = self.result_base_dir + result_name + '/d_loss.json'
    with open(g_loss_path) as g_loss_file:
      self.g_loss_list = json.loads(g_loss_file.read())
    with open(d_loss_path) as d_loss_file:
      self.d_loss_list = json.loads(d_loss_file.read())
      
  def generate(self, num_samples, seq_len):
    self.G.eval()
    with torch.no_grad():
      z = torch.randn(num_samples, seq_len, self.generator_input_dim).to(self.device)
      fake_data = self.G(z)
    return fake_data.cpu().numpy()