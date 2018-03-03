# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 2018

@author: Yue Peng, Ludan Zhang, Jiachen Zhang
"""
import pip
import os
import time
import sys
from collections import OrderedDict
pkgs = ['numpy', 'scipy', 'sklearn', 'pandas']
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


def authors():
    print("@authors: Yue Peng, Ludan Zhang, Jiachen Zhang\n")


class BrainD:

    def __init__(self, files):
        self._files = sorted(files)

    def fisher_z(self, row):
        return [*map(lambda x: 1 / 2 * np.log((1 + x) / (1 - x)) if x != 1.0 else 0., row)]

    def read_txt(self, file):
        return pd.read_table(file, sep=" ", header=None)

    def process(self, df):
        corr_mat = np.asarray(df.corr()).astype("float64").tolist()
        return [*map(self.fisher_z, corr_mat)]

    def scaling(self, df):
        return scale(df).tolist()

    def save_csv(self, name):
        np.savetxt("%s.csv" % (name), files_to_save[name], delimiter=",", fmt="%f")

    def main(self):
        global files_to_save
        files_to_save = OrderedDict()
        pool = ThreadPool(4)  # current cpu processes number
        train_dfs = pool.map(self.read_txt, self._files[0:410])
        test_dfs = pool.map(self.read_txt, self._files[410:820])
        Fs_train = np.array(pool.map(self.process, train_dfs)).reshape(410, 15, 15).astype("float64")
        Fs_test = np.array(pool.map(self.process, test_dfs)).reshape(410, 15, 15).astype("float64")
        Fs = np.concatenate((Fs_train, Fs_test), axis=0)
        files_to_save["Fn"], files_to_save["Fv"] = np.mean(Fs, axis=0), np.var(Fs, axis=0)

        files_to_save["Ftrain"], files_to_save["Ftest"] = np.mean(Fs_train, axis=0), np.mean(Fs_test, axis=0)
        Xs_train = np.array(pool.map(self.scaling, train_dfs)).reshape(-1, 15).astype("float64")
        Xs_test = np.array(pool.map(self.scaling, test_dfs)).reshape(-1, 15).astype("float64")
        files_to_save["Ctrain"], files_to_save["Ctest"] = np.cov(Xs_train.T), np.cov(Xs_test.T)
        files_to_save["U"], s, v = np.linalg.svd(Xs_train, full_matrices=False, compute_uv=True)
        files_to_save["G"] = np.dot(np.diag(s), v).astype("float64")
        UG = np.dot(files_to_save["U"], np.dot(np.diag(s), v)).astype("float64")
        files_to_save["CUG"] = np.cov(UG.T)
        files_to_save["CUGCtest"] = np.array(np.linalg.norm(files_to_save["CUG"] - files_to_save["Ctest"], ord="fro")).reshape(1, 1)
        files_to_save["CtrainCtest"] = np.array(np.linalg.norm(files_to_save["Ctrain"] - files_to_save["Ctest"], ord="fro")).reshape(1, 1)
        print("The closeness between matrix CUG and Ctest is    %.32f\nThe closeness between matrix Ctrain and Ctest is %.32f\n" % (files_to_save["CUGCtest"], files_to_save["CtrainCtest"]))
        print("Until here, elapsed time is %.2fs" % (time.time() - start))
        pool.map(self.save_csv, files_to_save.keys())
        pool.close()
        pool.join()


if __name__ == "__main__":
    print("HW2 started...\n")
    start = time.time()
    if os.path.basename(os.getcwd()) == "brainD15":
        authors()
        files = [s for s in os.listdir(os.getcwd()) if s.endswith(".txt")]
        hw2 = BrainD(files)
        hw2.main()
        print("HW2 was done!\nElapsed time is %.2fs" % (time.time() - start))
    else:
        sys.exit("You should move your python script into brainD15 folder.\n")
