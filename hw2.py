# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 2018

@author: Yue Peng, Ludan Zhang, Jiachen Zhang
"""
import pip
import os
import time
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
        from operator import add
        from functools import reduce


def authors():
    print("@authors: Yue Peng, Ludan Zhang, Jiachen Zhang\n")


class BrainD:

    def __init__(self, files):
        self._files = sorted(files)

    def fisher_z(self, row):
        return [*map(lambda x: 1 / 2 * np.log((1 + x) / (1 - x)) if x != 1.0 else 0., row)]

    def process(self, file):
        df = pd.read_table(file, sep=" ", header=None)
        corr_mat = np.asarray(df.corr()).tolist()
        return [*map(self.fisher_z, corr_mat)]

    def scaling(self, file):
        X = scale(pd.read_table(file, sep=" ", header=None)).tolist()
        return X

    def main(self):
        corr_new = [*map(self.process, self._files)]
        Fs = np.array(reduce(add, corr_new)).reshape(820, 15, 15)
        Fs = np.stack(Fs, axis=2)
        Fn, Fv = np.mean(Fs, axis=2), np.var(Fs, axis=2)
        np.savetxt("Fn.csv", Fn, delimiter=",")
        np.savetxt("Fv.csv", Fv, delimiter=",")
        train_files = self._files[0:410]
        test_files = self._files[410:820]
        corr_new_train = corr_new[0:410]
        corr_new_test = corr_new[410:820]
        Fs_train = np.array(reduce(add, corr_new_train)).reshape(410, 15, 15)
        Fs_test = np.array(reduce(add, corr_new_test)).reshape(410, 15, 15)
        Fs_train, Fs_test = np.stack(Fs_train, axis=2), np.stack(Fs_test, axis=2)
        Ftrain, Ftest = np.mean(Fs_train, axis=2), np.mean(Fs_test, axis=2)
        np.savetxt("Ftrain.csv", Ftrain, delimiter=",")
        np.savetxt("Ftest.csv", Ftest, delimiter=",")
        Xs_train = np.array(reduce(add, [*map(self.scaling, train_files)])).reshape(-1, 15)
        Xs_test = np.array(reduce(add, [*map(self.scaling, test_files)])).reshape(-1, 15)
        cov_Xs_train, cov_Xs_test = np.cov(Xs_train.T), np.cov(Xs_test.T)
        u, s, v = np.linalg.svd(Xs_train, full_matrices=False, compute_uv=True)
        g = np.dot(np.diag(s), v)
        UG = np.dot(u, np.dot(np.diag(s), v))
        cov_UG = np.cov(UG.T)
        dist_UG = np.linalg.norm(cov_UG - cov_Xs_test, ord="fro")
        dist_train = np.linalg.norm(cov_Xs_train - cov_Xs_test, ord="fro")
        print("The closeness between matrix CUG and Ctest is %.20f\nThe closeness between matrix Ctrain and Ctest is %.20f\n" % (dist_UG, dist_UG))
        print("Until here, elapsed time is %.2fs" % (time.time() - start))
        np.savetxt("U.csv", u, delimiter=",")
        np.savetxt("G.csv", g, delimiter=",")
        np.savetxt("CUG.csv", cov_UG, delimiter=",")
        np.savetxt("Ctrain.csv", cov_Xs_train, delimiter=",")
        np.savetxt("Ctest.csv", cov_Xs_test, delimiter=",")
        np.savetxt("CUGCtest.csv", np.array(dist_UG).reshape(1, 1), delimiter=",")
        np.savetxt("CtrainCtest.csv", np.array(dist_train).reshape(1, 1), delimiter=",")
        print("HW2 was done!\nElapsed time is %.2fs" % (time.time() - start))


if __name__ == "__main__":
    print("HW2 started...\n")
    start = time.time()
    if os.path.basename(os.getcwd()) == "brainD15":
        authors()
    else:
        print("You should move your python script into brainD15 folder.\n")
    # extract only .txt file in current path
    files = [s for s in os.listdir(os.getcwd()) if s.endswith(".txt")]
    hw2 = BrainD(files)
    hw2.main()
