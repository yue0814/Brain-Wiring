# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 2018

@author: Yue Peng, Ludan Zhang, Jiachen Zhang
"""
import pip
import os
import time
import sys
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
        from collections import OrderedDict


class BrainD:

    def __init__(self, files):
        self._files = sorted(files)

    def authors(self):
        print("@authors: Yue Peng, Ludan Zhang, Jiachen Zhang\n")

    def read_txt(self, file):
        return pd.read_table(file, sep=" ", header=None)

    def process(self, df):
        corr_mat = np.asarray(df.corr()).astype("float64")
        np.putmask(corr_mat, corr_mat == 1.0, 0.)
        return (1 / 2 * np.log(np.divide(1 + corr_mat, 1 - corr_mat))).tolist()

    def scaling(self, df):
        return scale(df).tolist()

    def save_csv(self, name):
        np.savetxt("%s.csv" % (name), files_to_save[name], delimiter=",", fmt="%f")

    def main(self):
        global files_to_save
        files_to_save = OrderedDict()
        pool = ThreadPool(os.cpu_count())  # current cpu processes number
        train_dfs = pool.map(self.read_txt, self._files[0:410])
        test_dfs = pool.map(self.read_txt, self._files[410:820])
        Fs_train = np.array(pool.map(self.process, train_dfs), dtype=np.float64).reshape(410, 15, 15)
        Fs_test = np.array(pool.map(self.process, test_dfs), dtype=np.float64).reshape(410, 15, 15)
        Fs = np.concatenate((Fs_train, Fs_test), axis=0)
        files_to_save["Fn"], files_to_save["Fv"] = np.mean(Fs, axis=0), np.var(Fs, axis=0)

        files_to_save["Ftrain"], files_to_save["Ftest"] = np.mean(Fs_train, axis=0), np.mean(Fs_test, axis=0)
        Xs_train = np.array(pool.map(self.scaling, train_dfs), dtype=np.float64).reshape(-1, 15)
        Xs_test = np.array(pool.map(self.scaling, test_dfs), dtype=np.float64).reshape(-1, 15)
        files_to_save["Ctrain"], files_to_save["Ctest"] = np.cov(Xs_train.T), np.cov(Xs_test.T)
        files_to_save["U"], s, v = np.linalg.svd(Xs_train, full_matrices=False, compute_uv=True)
        files_to_save["G"] = np.dot(np.diag(s), v)
        UG = np.dot(files_to_save["U"], np.dot(np.diag(s), v))
        files_to_save["CUG"] = np.cov(UG.T)
        files_to_save["CUGCtest"] = np.array(np.linalg.norm(files_to_save["CUG"] - files_to_save["Ctest"], ord="fro")).reshape(1, 1)
        files_to_save["CtrainCtest"] = np.array(np.linalg.norm(files_to_save["Ctrain"] - files_to_save["Ctest"], ord="fro")).reshape(1, 1)
        print("The closeness between matrix CUG and Ctest is    %.32f\nThe closeness between matrix Ctrain and Ctest is %.32f\n" % (files_to_save["CUGCtest"], files_to_save["CtrainCtest"]))
        pool.map(self.save_csv, files_to_save.keys())
        pool.close()
        pool.join()


if __name__ == "__main__":
    print("HW2 started...\n")
    start = time.time()
    if os.path.basename(os.getcwd()) == "brainD15":
        files = [s for s in os.listdir(os.getcwd()) if s.endswith(".txt")]
        hw2 = BrainD(files)
        hw2.authors()
        hw2.main()
        print("HW2 was done!\nElapsed time is %.2fs" % (time.time() - start))
    else:
        sys.exit("You should move your python script into brainD15 folder.\n")
