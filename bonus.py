# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 2018

Bonus script of hw2

@author: Yue Peng, Ludan Zhang, Jiachen Zhang
"""
import os
import sys
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool


def scaling(file):
    return scale(pd.read_table(file, sep=" ", header=None)).tolist()

Xs_train = np.array(map(scaling, files[0:410])).reshape(-1, 15).astype("float64")
Xs_test = np.array(pool.map(scaling, files[410:820])).reshape(-1, 15).astype("float64")