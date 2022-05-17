from rsnn_spinn import *
import spynnaker8 as sim

s1 = RSNN_SPyNN(sim, timestep =1.0, modelname='rnn_lsm_512/', num_to_test= 100, tsf=5.0, npc=10)    
s1.run()
s1.plot_save()
s1.sim.end()