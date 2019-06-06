#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:52:22 2019

@author: dan
"""

import numpy as np

def compute_chi_shuffle(mean_sweep_events,sweep_conditions,num_shuffles=1000):

    (num_sweeps,num_cells) = np.shape(mean_sweep_events) 
    
    expected = compute_expected(mean_sweep_events,sweep_conditions)
    observed = compute_observed(mean_sweep_events,sweep_conditions)
    chi_actual = compute_chi(observed,expected)
    
    chi_shuffle = np.zeros((num_cells,num_shuffles))
    for ns in range(num_shuffles):
        #print 'shuffle ' + str(ns+1) + ' of ' + str(num_shuffles)
        
        shuffle_sweeps = np.random.choice(num_sweeps,size=(num_sweeps,))
        shuffle_sweep_events = mean_sweep_events[shuffle_sweeps]
        
        shuffle_expected = compute_expected(shuffle_sweep_events,sweep_conditions)
        shuffle_observed = compute_observed(shuffle_sweep_events,sweep_conditions)
        
        chi_shuffle[:,ns] = compute_chi(shuffle_observed,shuffle_expected)
    
    p_vals = np.mean(chi_actual.reshape(num_cells,1)<chi_shuffle,axis=1)
    
    return p_vals

def compute_observed(mean_sweep_events,sweep_conditions):

    (num_sweeps,num_conditions) = np.shape(sweep_conditions)
    num_cells = np.shape(mean_sweep_events)[1]   
    
    observed_mat = (mean_sweep_events.T).reshape(num_cells,num_sweeps,1) * sweep_conditions.reshape(1,num_sweeps,num_conditions)
    observed = np.sum(observed_mat,axis=1)
    
    return observed
    
def compute_expected(mean_sweep_events,sweep_conditions):   
    
    num_conditions = np.shape(sweep_conditions)[1]
    num_cells = np.shape(mean_sweep_events)[1]
    
    sweeps_per_condition = np.sum(sweep_conditions,axis=0)
    events_per_sweep = np.mean(mean_sweep_events,axis=0)
    
    expected = sweeps_per_condition.reshape(1,num_conditions) * events_per_sweep.reshape(num_cells,1) 
    
    return expected

def compute_chi(observed,expected):

    chi = (observed - expected) ** 2 /expected
    chi = np.where(expected>0,chi,0.0)  
    return np.sum(chi,axis=1)