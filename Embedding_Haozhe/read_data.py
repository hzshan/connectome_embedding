#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:52:21 2024

@author: mbeiran
"""
import numpy as np

file = np.load('params_netRing_init.npz')#, JJ, gg, bb, hh0, wI, wOut, alpha_, si_)
J0 = file['arr_0']
g0 = file['arr_1']
Jeff0 = J0.dot(np.diag(g0)) # this is the effective connectivity


ori0_s = np.linspace(-np.pi*0.5, np.pi*0.5, 8)
Jeffs = []

for ori0_ in ori0_s:
    file = np.load('params_netRing_ori_'+str(ori0_)[0:6]+'_final.npz')
    Jfs = file['arr_0']
    gfs = file['arr_1']
    Jeffs.append(Jfs.dot(np.diag(gfs)))
