# -*- coding: utf-8 -*-
"""
Created on Mon May 16 18:14:11 2022

@author: alberto
"""

import torch
import torch.nn as nn
from torch import Tensor
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        thresh = 0.3       
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        thresh = 0.3
        lens = 0.25        
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float() / (2 * lens)

class RSNN_Model(nn.Module):

    def __init__(self, num_hidden = 256, win= 25, batch_size=20, device='cpu'):
        super(RSNN_Model, self).__init__()
        
        self.act_fun = ActFun.apply
        
        self.thresh = 0.3
        self.num_input = 34*34*2
        self.num_hidden = num_hidden
        self.num_output = 10
        self.tau_m = torch.Tensor([9.4912])
        self.batch_size = batch_size
        self.device = device
        self.win = win
        self.fc_ih = nn.Linear(self.num_input, self.num_hidden, bias= False)
        self.fc_hh = nn.Linear(num_hidden, self.num_hidden, bias= False)
        self.fc_ho = nn.Linear(num_hidden, self.num_output, bias= False)
        self.num_samples = 60000
        
        self.epoch = 0
        self.acc = list()
        self.train_loss = list()
        self.test_loss = list()

       
    def forward(self, input):

        h_mem = h_spike = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        o_mem = o_spike = o_sumspike = torch.zeros(self.batch_size, self.num_output, device=self.device)

        for step in range(self.win):

            x = input[:, step, :]
            i_spike = x.view(self.batch_size, -1)

            h_mem, h_spike = self.mem_update_rnn(i_spike, h_spike, h_mem)            
            o_mem, o_spike = self.mem_update(h_spike, o_spike, o_mem)

            o_sumspike = o_sumspike + o_spike

        outputs = o_sumspike / (self.win)

        return outputs        

    def mem_update(self, i_spike, o_spike, mem):
        alpha = torch.exp(-1. / self.tau_m).to(self.device)
        mem = mem * alpha * (1 - o_spike) + self.fc_ho(i_spike) 
        o_spike = self.act_fun(mem)
        mem = mem*(mem<self.thresh)
        return mem, o_spike

    def mem_update_rnn(self, i_spike, o_spike, mem):
        alpha = torch.exp(-1. / self.tau_m).to(self.device)
        a = self.fc_ih(i_spike) # process spikes from input
        b = self.fc_hh(o_spike) # process recurrent spikes
        c = mem * alpha * (1-o_spike)
        mem = a + b + c
        o_spike = self.act_fun(mem)
        mem = mem*(mem<self.thresh)
        return mem, o_spike
    
    def freeze_parameters(self):
        '''
        make input-to-hidden and hidden-to-hidden connections untrainable
        '''
        for i, parameter in enumerate(self.parameters()):
            print(parameter.shape)
            if i<2:
                parameter.requires_grad = False
                print('Frozen')   

    def quantize_weights(self, bits):
        
        def reduce_precision(weights, bits):
            scale = 1.001*(1+bits)*(weights.max()-weights.min())/(2*bits+3)
            m = scale*torch.round((weights/scale)*2**bits)/(2**bits)
            return m   

        with torch.no_grad():
            self.fc_hh.weight.data = torch.nn.Parameter(reduce_precision(self.fc_hh.weight.data, bits-1))
            self.fc_ih.weight.data = torch.nn.Parameter(reduce_precision(self.fc_ih.weight.data, bits-1))
            #self.fc_ho.weight.data = torch.nn.Parameter(reduce_precision(self.fc_ho.weight.data, 1))                  
                
    def train_step(self, train_loader=None, optimizer=None, criterion=None):
        
        total_loss_train = 0
        running_loss = 0
        total = 0
        
        num_iter = self.num_samples // self.batch_size 
        
        for i, (images, labels) in enumerate(train_loader):
            self.zero_grad()
            optimizer.zero_grad()
            images = images.float().to(self.device)
            labels = labels.float().to(self.device)
            outputs = self(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            total_loss_train += loss.item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()

            if (i + 1) % int(num_iter/3.0) == 0:
                print('Step [%d/%d], Loss: %.5f'
                      % (i + 1, self.num_samples // self.batch_size, running_loss))
                running_loss = 0
                
        self.epoch = self.epoch + 1
        self.train_loss.append([self.epoch, total_loss_train / total]) 

    def lr_scheduler(self, optimizer, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.98 every lr_decay_epoch epochs."""

        if self.epoch % lr_decay_epoch == 0 and self.epoch > 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.98

        return optimizer            
        
    def test(self, test_loader = None, criterion=None):
        
        self.save_model()
        
        correct = 0
        total = 0
        total_loss_test = 0
        
        snn_cpu = type(self)() # copy of self, doing this to always evaluate on cpu
        snn_cpu.load_model('rsnn', batch_size= self.batch_size)
        
        for images, labels in test_loader:
            images = images.float()
            labels = labels.float()
            outputs = snn_cpu(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            _, reference = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == reference).sum()
            total_loss_test += loss.item() 
            
        acc = 100. * float(correct) / float(total)
        
        if self.acc == []:
            self.acc.append([self.epoch, acc]) 
            self.test_loss.append([self.epoch, total_loss_test / total])
        else:
            if self.acc[-1][0] < self.epoch:
                self.acc.append([self.epoch, acc]) 
                self.test_loss.append([self.epoch, total_loss_test / total])                          
                  
        print('Test Accuracy of the model on the test samples: %.3f' % (acc))    

    def plot_weights(self, w, mode='histogram', crop_size = None ):
        '''
        plots weights as mode=histogram or as mode=matrix
        '''
        
        name='weight distribution'
        
        if w == 'hh':
            w = self.fc_hh
            name = 'hidden-to-hidden weight distribution'
        elif w == 'ih':
            w = self.fc_ih
            name = 'input-to-hidden weight distribution'
        elif w == 'ho':     
            w = self.fc_ho          
            name = 'hidden-to-output weight distribution'        
        
        w = w.weight.data.cpu().numpy()    
        vmin = np.min(w)
        vmax = np.max(w)

        if crop_size is not None:
            w = w[:crop_size,:crop_size]

        if mode=='histogram':
            if self.device.type == 'cpu':
                w = list(w.reshape(1,-1)[0])
                n, bins, fig = plt.hist(w, bins=200)
            else:
                fig = sns.histplot(w.reshape(1,-1)[0], bins = 200)
            plt.xlabel('weight', fontsize=14)
            plt.ylabel('frequency', fontsize=14)
            plt.xlim(vmin*1.5, vmax*1.5)
            plt.title(name, fontsize=16)
            return fig
        elif mode=='matrix':
            fig = plt.figure(figsize=(10,10))
            c= 'RdBu'
            plt.imshow(w, cmap=c, vmin=vmin, vmax=vmax)
            plt.xlabel('input', fontsize=14)
            plt.ylabel('output', fontsize=14)
            plt.title('weights', fontsize=16)
            plt.colorbar()
            return fig          
        
    def plot_loss(self):
        
        test_loss = np.array(self.test_loss)
        train_loss = np.array(self.train_loss)        
        fig = plt.figure()
        plt.plot(train_loss[:,0], train_loss[:,1], label ='train loss')
        plt.plot(test_loss[:,0], test_loss[:,1], label = 'test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        
        return fig   
            
    def save_model(self, modelname = 'rsnn'):
        '''
        save model in the pytorch format to /checkpoint
        '''
        state = {
            'net': self.state_dict(),
            'epoch': self.epoch,
            'acc_record': self.acc,
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
            'num_hidden': self.num_hidden,
            'thresh':self.thresh,
            'win': self.win
        }         
        
        torch.save(state, './checkpoint/' + modelname,  _use_new_zipfile_serialization=False)

    def load_model(self, modelname=None, location = '', batch_size=256, device='cpu'):
        params = torch.load('./checkpoint'+location+'/'+modelname, map_location=torch.device('cpu'))
        
        self.__init__(params['num_hidden'], params['win'], batch_size, device)
        self.load_state_dict(params['net'])
        self.acc = params['acc_record'] 
        self.train_loss = params['train_loss']
        self.test_loss = params['test_loss']           
        
    def save_to_numpy(self, directory = None):
        '''
        saves weights to numpy files that will be later loaded by PyNN / SpyNNaker
        '''
        layers_location = './../spinnaker/models/' + directory

        if not os.path.isdir(layers_location):
            os.mkdir(layers_location)

        snn_state_dict = self.state_dict()
        
        for k in snn_state_dict:
            np.savez(layers_location+'/'+k,snn_state_dict[k].data.cpu().numpy())
                