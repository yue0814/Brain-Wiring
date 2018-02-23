# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 2018

@author: Yue Peng, Ludan Zhang, Jiachen Zhang
"""
import pip
import os
import sys
pkgs = ['numpy', 'scipy', 'sklearn', 'pandas']
for package in pkgs:
    try:
        import package
    except ImportError:
        pip.main(['install', package])

if os.path.basename(os.getcwd()) == "brainD15":
    import pandas as pd
    import numpy as np
    import time
    from collections import OrderedDict
    from sklearn.preprocessing import scale

else:
    sys.eixt("You should move your python script into brainD15 folder.\n")


def authors():
    print("@authors: Yue Peng, Ludan Zhang, Jiachen Zhang\n")


class BrainD:

    def __init__(self, files):
        self._files = files

    def fisher_z(self, col):
        return [*map(lambda x: 1 / 2 * np.log((1 + x) / (1 - x)) if x != 1.0 else 0., col)]

    def transform(self, matrix):
        return np.array([*map(self.fisher_z, matrix)])

    def extract(self):
        sorted_files = sorted(self._files)  # sorted by subject IDs
        train_ids = sorted_files[0:410]
        test_ids = sorted_files[410:820]

        corr_dict = OrderedDict()  # remain the order in Dict
        for i, v in enumerate(sorted_files):
            dat = pd.read_table(v, sep=" ", header=None)
            corr_dict[v] = self.transform(np.array(dat.corr()))

        Xs_train, Xs_test = [scale(pd.read_table(v, sep=" ", header=None)) for v in train_ids], [
            scale(pd.read_table(v, sep=" ", header=None)) for v in test_ids]

        Fs_train, Fs_test = [corr_dict[v] for v in train_ids], [
            corr_dict[v] for v in test_ids]
        Xs_train, Xs_test = np.array(Xs_train), np.array(Xs_test)
        Fs_train, Fs_test = np.stack(
            Fs_train, axis=2), np.stack(Fs_test, axis=2)
        Fn_train, Fn_test = np.mean(
            Fs_train, axis=2), np.mean(Fs_test, axis=2)
        Xs_train, Xs_test = np.reshape(
            Xs_train, (-1, 15)), np.reshape(Xs_test, (-1, 15))
        print("Finished extract all the matrices\n")
        return Fn_train, Fn_test, Xs_train, Xs_test, corr_dict

    def average_var(self, corr_dict):
        Fs = [v for _, v in corr_dict.items()]
        # concatenate all 820 matrices into 3-d array
        Fs = np.stack(Fs, axis=2)
        Fn, Fv = np.mean(Fs, axis=2), np.var(Fs, axis=2)
        return Fn, Fv

    def svd(self, Xs_train):
        # get SVD components from train matrix
        u, s, v = np.linalg.svd(
            Xs_train, full_matrices=False, compute_uv=True)
        g = np.dot(np.diag(s), v)
        UG = np.dot(u, np.dot(np.diag(s), v))
        return u, g, UG
        # numpy will transpose matrix first before calculate cov

    def cov(self, Xs_train, Xs_test, UG):
        cov_UG = np.cov(UG.T)
        cov_Xs_train, cov_Xs_test = np.cov(Xs_train.T), np.cov(Xs_test.T)
        return cov_UG, cov_Xs_train, cov_Xs_test

    def norm(self, cov_UG, cov_Xs_train, cov_Xs_test):
        dist_UG = np.linalg.norm(cov_UG - cov_Xs_test, ord="fro")
        dist_train = np.linalg.norm(cov_Xs_train - cov_Xs_test, ord="fro")
        print("The closeness between matrix CUG and Ctest is %.13f\nThe closeness between matrix Ctrain and Ctest is %.13f\n" % (
            dist_UG, dist_UG))
        print("Until here, elapsed time is %.2fs\n" % (time.time() - start))
        return dist_UG, dist_train

    def save_csv(self, Fn, Fv, Fn_train, Fn_test, u, g, cov_UG, cov_Xs_train, cov_Xs_test, dist_UG, dist_train):
        mat_dict = {"Fn": Fn, "Fv": Fv, "Ftrain": Fn_train, "Ftest": Fn_test, "U": u, "G": g, "CUG": cov_UG,
                    "Ctrain": cov_Xs_train, "Ctest": cov_Xs_test, "CUGCtest": dist_UG, "CtrainCtest": dist_train}
        for k, v in mat_dict.items():
            np.savetxt("%s.csv" % (k), v, delimiter=",")
        print("Finished saving all matrices into csv file.\n")

    def main(self):
        Fn_train, Fn_test, Xs_train, Xs_test, corr_dict = self.extract()
        Fn, Fv = self.average_var(corr_dict)
        u, g, UG = self.svd(Xs_train)
        cov_UG, cov_Xs_train, cov_Xs_test = self.cov(Xs_train, Xs_test, UG)
        dist_UG, dist_train = self.norm(cov_UG, cov_Xs_train, cov_Xs_test)
        dist_UG, dist_train = np.array(dist_UG).reshape(
            (1, 1)), np.array(dist_train).reshape((1, 1))
        self.save_csv(Fn, Fv, Fn_train, Fn_test, u, g, cov_UG, cov_Xs_train, cov_Xs_test, dist_UG, dist_train)


if __name__ == "__main__":
    print("HW2 started...\n")
    start = time.time()
    authors()
    # extract only .txt file in current path
    files = [s for s in os.listdir(os.getcwd()) if s.endswith(".txt")]
    hw2 = BrainD(files)
    hw2.main()
    print("HW2 was done!\nElapsed time is %.2fs" % (time.time() - start))

