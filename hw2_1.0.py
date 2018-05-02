# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 2018

@author: Yue Peng, Ludan Zhang, Jiachen Zhang
"""
import pip
import os
import time
import sys
import codecs
from collections import OrderedDict
pkgs = ["numpy", "scipy", "sklearn", "pandas"]
for package in pkgs:
    try:
        import package
    except ImportError:
        pip.main(['install', package])
    finally:
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import scale
        from multiprocessing.dummy import Pool as ThreadPool

files = sorted([s for s in os.listdir(os.getcwd()) if s.endswith(".txt")])


def authors():
    print("@authors: Yue Peng, Ludan Zhang, Jiachen Zhang\n")


def preprocess(files):
    for _, v in enumerate(files):
        mat = np.loadtxt(v)
        cor_mat = np.corrcoef(mat, rowvar=0)
