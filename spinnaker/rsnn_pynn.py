#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 12 2022

@author: alberto
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import json
import os

class RSNN_PyNN():
    
    def __init__(self, sim, timestep =1.0, dataset='mnist', modelname='256/', num_to_test= 20, record_hidden = True):
        
        self.sim = sim
        self.name = dataset +'_' +modelname
        self.timestep = timestep
        self.num_to_test = num_to_test
        self.sim.setup(timestep, min_delay=1.0, max_delay=2.0)
       
        weights = []
        weights_path = './../notebooks/checkpoint/'+self.name
        weights.append(np.load(weights_path+'fc_ih.weight.npz')['arr_0'])
        weights.append(np.load(weights_path+'fc_hh.weight.npz')['arr_0'])
        weights.append(np.load(weights_path+'fc_ho.weight.npz')['arr_0'])     
        
        self.n_h = len(weights[0])
             
        if dataset=='shd':
            self.win = 50
            self.next_sample_delay = 30
            self.sample_size = 700
            input_file = "./input_spiketrains/heidelberg_{}_delay_{}.json".format(self.num_to_test, self.next_sample_delay)
            v_th_h = 0.77
            off = 0.34   
            inicio = 0
        
        else:
            self.win = 25
            self.next_sample_delay = 25
            self.sample_size = 34*34*2
            input_file = "./input_spiketrains/nmnist_{}_delay_{}.json".format(self.num_to_test, self.next_sample_delay)
            v_th_h = 0.55
            off = 0.3

        self.inicio = 0    
        
        self.total_duration = self.win+self.next_sample_delay

        with open(input_file) as jsonfile:
            data = json.load(jsonfile)

        self.spike_times = data['input_spikes']
        self.label = np.array(data['label'])

        delay = 1.0
        v_th = 0.3
        q=0.1
        tau_syn = 0.01/q        
        e_rev = 10000
        
        cellvalues = {'v': 0.0}

        #cm = e_rev/181.0
        cm = e_rev/345.0
        #cm = e_rev/333.0

        cellparams = {'v_thresh': v_th, 'v_reset': 0.0, 'v_rest': 0.0, 'e_rev_E': e_rev, 'e_rev_I': -e_rev, 'i_offset': 0.0,
                      'cm': cm/q, 'tau_m': 0.8305, 'tau_syn_E': tau_syn, 'tau_syn_I': tau_syn, 'tau_refrac': 0.0}

        self.liquid_pop = self.sim.Population(len(weights[0]), self.sim.IF_cond_exp(**cellparams))
        self.output_pop = self.sim.Population(len(weights[-1]), self.sim.IF_cond_exp(**cellparams))    

        i_o = 0.7
        cellparams_h = {'v_thresh': v_th_h, 'v_reset': 0.0, 'v_rest': 0.0, 'e_rev_E': 0.1, 'e_rev_I': -v_th, 'i_offset': i_o,
                       'cm': 50.0, 'tau_m': 100.0, 'tau_syn_E': tau_syn, 'tau_syn_I': tau_syn, 'tau_refrac': 0.0}


        # h_pop = sim.Population(len(weights[0]), sim.IF_cond_exp(**cellparams_h))

        # sim.Projection(h_pop,liquid_pop, connector=sim.OneToOneConnector(), synapse_type=sim.StaticSynapse(weight=5.0, delay=dt),
        #                     receptor_type='inhibitory')

        h_pop = self.sim.Population(1, sim.IF_cond_exp(**cellparams_h))
        

        self.sim.Projection(h_pop,self.liquid_pop, connector=sim.AllToAllConnector(), synapse_type=sim.StaticSynapse(weight=5.0, delay=1.0),
                            receptor_type='inhibitory')

        self.set_input_population()
        self.connect(weights, delay=1.0)

        self.liquid_pop.initialize(**cellvalues)
        self.output_pop.initialize(**cellvalues)    
        h_pop.initialize(v=off) 

        self.output_pop.record(['spikes'])
        
        if record_hidden:
            self.liquid_pop.record(['spikes'])
        
    def set_input_population(self, spk_per_timestep=None): 
        
        if spk_per_timestep is not None:
            self.spike_times = [[*np.random.choice(np.arange(0,self.sample_size), spk_per_timestep, replace=False)] for x in range(self.total_duration*self.num_to_test)]
        
        self.input_pop = self.sim.Population(self.sample_size, self.sim.SpikeSourceArray())
        self.input_pop.set(spike_times=self.spike_times)     

    def run(self):

        self.sim.run(self.total_duration*self.num_to_test / self.timestep)

        spiketrains_output = self.output_pop.get_data().segments[-1].spiketrains

        ## PREDICTIONS

        out_spks_nope = self.spk_to_array(spiketrains_output)
        self.out_spks = np.zeros((len(spiketrains_output),self.win*self.num_to_test))

        preds = []
        for x in range(self.num_to_test):
            a = x*self.win
            b = x*(self.win+self.next_sample_delay)

            self.out_spks[:,a:a+self.win] = out_spks_nope[:,b:b+self.win]

            preds.append(self.out_spks[:,a:a+self.win].sum(axis=1).argmax())

        print(self.label[self.inicio:self.inicio+self.num_to_test].argmax(axis=1))

        print(np.array(preds))

        acc = np.float(np.sum(np.array(preds) == self.label[self.inicio:self.inicio+self.num_to_test].argmax(axis=1)))
        print('accuracy: ' +str(100*(acc/self.num_to_test)) + '%')   
        
        self.sim.end()
        
    def project(self, weights, delay):           
        inh_synapses = []
        exc_synapses = []

        for i in range(weights.shape[1]):
            for j in range(weights.shape[0]):
                if float(weights[j, i])<0.0:
                    inh_synapses.append([i, j, -1.0*weights[j, i], delay])
                else:
                    exc_synapses.append([i, j, weights[j, i], delay]) 
                    
        return  inh_synapses, exc_synapses

    def connect(self, weights, delay):
     
        pops = [self.input_pop, self.liquid_pop, self.liquid_pop, self.output_pop]

        for layer in range(len(weights)):
            
            w = weights[layer]
            
            inh_synapses, exc_synapses = self.project(w, delay)

            self.sim.Projection(pops[layer],pops[layer+1], connector=self.sim.FromListConnector(inh_synapses), receptor_type='inhibitory')
            self.sim.Projection(pops[layer],pops[layer+1], connector=self.sim.FromListConnector(exc_synapses), receptor_type='excitatory')     

    def spk_count(self, spiketrain, start, end):

        spikecounts = np.zeros(len(spiketrain))
        for neuron, spk in enumerate(spiketrain):
            spikecounts[neuron] =  len(list(filter(lambda x: x>=start and x<end, spk)))

        return spikecounts.argmax()

    def spk_to_array(self, spiketrain):
        n = len(spiketrain)
        t = self.total_duration*self.num_to_test
        spks = np.zeros((n,t))
        for neu_idx, spk in enumerate(spiketrain):
            for time in spk:
                spks[neu_idx, int(time)-1] = 1
        return spks   

    def plot_save(self):

        print('plotting....')

        plots_folder = './results/spinn_'+self.name
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        #membranes_output = output_pop.get_data().segments[-1].filter(name='v')[0]
        spiketrains_liquid = self.liquid_pop.get_data().segments[-1].spiketrains
        #membranes_liquid = liquid_pop.get_data().segments[-1].filter(name='v')[0]
        #spiketrains_input = input_pop.get_data().segments[-1].spiketrains    
        #spiketrains_h = h_pop.get_data().segments[-1].spiketrains
        #membranes_h = h_pop.get_data().segments[-1].filter(name='v')[0]  
        #g_e = np.array(liquid_pop.get_data().segments[-1].filter(name='gsyn_exc')[0])
        #g_i = np.array(liquid_pop.get_data().segments[-1].filter(name='gsyn_inh')[0])

        # fig = plt.figure('input')
        # plt.eventplot(spiketrains_input, linelengths=0.7, colors='k', label='pre_spikes') 
        # plt.ylabel('Neuron index')
        # plt.xlabel('Time (ms)') 
        # for x in range(num_to_test-1):
        #     plt.vlines(4+total_duration*(x+1), -1, sample_size, 'g', 'dashed')
        # #fig.savefig('nest_last.png',dpi=300)
        # plt.show()

        fig = plt.figure('hidden', figsize=(15,10))
        plt.eventplot(spiketrains_liquid, linelengths=0.7, colors='k', label='hidden_layer_spikes') 
        plt.tick_params(labelsize=20)
        plt.ylabel('Neuron index', fontsize= 30)
        plt.xlabel('Time (ms)', fontsize= 30) 
        for x in range(self.num_to_test-1):
            plt.vlines(4+self.total_duration*(x+1), -1, self.n_h, 'g', 'dashed')
        plt.savefig(plots_folder+'/hidden_spk.png',dpi=300)
        #plt.savefig(plots_folder+'/hidden_spk.svg',dpi=600)
        #plt.savefig(plots_folder+'/hidden_spk.pdf',dpi=600)
        plt.show()

        #mems_nope = np.array(membranes_liquid)
        spks_nope = self.spk_to_array(spiketrains_liquid)

        #mems = np.zeros((win*num_to_test,len(spiketrains_liquid)))
        spks = np.zeros((len(spiketrains_liquid),self.win*self.num_to_test))

        for x in range(self.num_to_test):
            a = x*self.win
            b = x*(self.win+self.next_sample_delay)
            #mems[a:a+win,:] = mems_nope[b:b+win,:]
            spks[:,a:a+self.win] = spks_nope[:,b:b+self.win]

        #np.save(plots_folder+'/mems.npy', mems.T)
        np.save(plots_folder+'/spks.npy', spks)

        #means_n = spks.mean(axis=1)
        means_t = spks.sum(axis=0)

        plotname = '{}'.format(self.name)
        plt.figure(figsize=(15,10))
        plt.title('Recurrent layer activity per timestep, ' + plotname)
        plt.plot(means_t, label='spikes')
        #plt.plot(mems.mean(axis=1), label='avg membrane potential')
        plt.xlabel('Time (ms)')
        plt.legend()    
        plt.savefig(plots_folder+'/mems_t.png',dpi=300)
    
class RSNN_PyNN_split(RSNN_PyNN):
    
    def set_input_population(self):    
        
        self.n_splits = 2
        
        self.ipn = int(self.sample_size/self.n_splits)
        
        self.input_pops = [self.sim.Population(self.ipn, self.sim.SpikeSourceArray()) for x in range(self.n_splits)]
        
        for pop in range(self.n_splits):
            self.input_pops[pop].set(spike_times=self.spike_times[self.ipn*pop:self.ipn*pop+self.ipn])
        
    def connect(self, weights, delay):
     
        pops = [None, self.liquid_pop, self.liquid_pop, self.output_pop]
        
        for pop in range(self.n_splits):
                                                  
            w=weights[0][:,self.ipn*pop:self.ipn*pop+self.ipn]

            inh_synapses, exc_synapses = self.project(w, delay)

            self.sim.Projection(self.input_pops[pop],self.liquid_pop, connector=self.sim.FromListConnector(inh_synapses), receptor_type='inhibitory')
            self.sim.Projection(self.input_pops[pop],self.liquid_pop, connector=self.sim.FromListConnector(exc_synapses), receptor_type='excitatory')           
          
        for layer in range(1,len(weights)):

            w = weights[layer]
            
            inh_synapses, exc_synapses = self.project(w, delay)

            self.sim.Projection(pops[layer],pops[layer+1], connector=self.sim.FromListConnector(inh_synapses), receptor_type='inhibitory')
            self.sim.Projection(pops[layer],pops[layer+1], connector=self.sim.FromListConnector(exc_synapses), receptor_type='excitatory')    
            