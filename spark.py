# -*- coding = utf-8 -*-
"""
    Spark version of hw2
"""
import os
import time
import numpy as np
import pandas as pd
from operator import add
from sklearn.preprocessing import scale
from pyspark import SparkContext, SparkConf


def authors():
    print("@authors: Yue Peng\n")


def main(sc):
    files = [s for s in os.listdir(os.getcwd()) if s.endswith(".txt")]
    rdd = sc.parallelize(files)
    # sorted the files
    rdd = rdd.sortBy(lambda x: x)

    def fisher_z(row):
        return [*map(lambda x: 1 / 2 * np.log((1 + x) / (1 - x)) if x != 1.0 else 0., row)]

    def process(file):
        df = pd.read_table(file, sep=" ", header=None)
        corr_mat = np.asarray(df.corr()).tolist()
        corr_new = [*map(fisher_z, corr_mat)]
        return corr_new

    # problem 1
    corr_new = rdd.map(process)
    # problem 2
    Fs = np.array(corr_new.reduce(add)).reshape(820, 15, 15)
    Fs = np.stack(Fs, axis=2)
    Fn, Fv = np.mean(Fs, axis=2), np.var(Fs, axis=2)
    np.savetxt("Fn.csv", Fn, delimiter=",")
    np.savetxt("Fv.csv", Fv, delimiter=",")
    # problem 3
    train_rdd = rdd.filter(lambda x: True if float(x[3:9]) <= 211316 else False)
    test_rdd = rdd.filter(lambda x: True if float(x[3:9]) > 211316 else False)
    corr_new_train = train_rdd.map(process)
    corr_new_test = test_rdd.map(process)
    Fs_train = np.array(corr_new_train.reduce(add)).reshape(410, 15, 15)
    Fs_test = np.array(corr_new_test.reduce(add)).reshape(410, 15, 15)
    Fs_train, Fs_test = np.stack(Fs_train, axis=2), np.stack(Fs_test, axis=2)
    Ftrain, Ftest = np.mean(Fs_train, axis=2), np.mean(Fs_test, axis=2)
    np.savetxt("Ftrain.csv", Ftrain, delimiter=",")
    np.savetxt("Ftest.csv", Ftest, delimiter=",")

    # problem 4
    def scaling(file):
        X = scale(pd.read_table(file, sep=" ", header=None)).tolist()
        return X

    Xs_train = np.array(train_rdd.map(scaling).reduce(add)).reshape(-1, 15)
    Xs_test = np.array(test_rdd.map(scaling).reduce(add)).reshape(-1, 15)
    cov_Xs_train, cov_Xs_test = np.cov(Xs_train.T), np.cov(Xs_test.T)
    u, s, v = np.linalg.svd(Xs_train, full_matrices=False, compute_uv=True)
    g = np.dot(np.diag(s), v)
    UG = np.dot(u, np.dot(np.diag(s), v))
    cov_UG = np.cov(UG.T)
    dist_UG = np.array(np.linalg.norm(cov_UG - cov_Xs_test, ord="fro")).reshape(1, 1)
    dist_train = np.array(np.linalg.norm(cov_Xs_train - cov_Xs_test, ord="fro")).reshape(1, 1)
    print("The closeness between matrix CUG and Ctest is %f\nThe closeness between matrix Ctrain and Ctest is %f\n" % (dist_UG, dist_UG))
    print("Until here, elapsed time is %.2fs\n" % (time.time() - start))
    np.savetxt("U.csv", u, delimiter=",")
    np.savetxt("G.csv", g, delimiter=",")
    np.savetxt("CUG.csv", cov_UG, delimiter=",")
    np.savetxt("Ctrain.csv", cov_Xs_train, delimiter=",")
    np.savetxt("Ctest.csv", cov_Xs_test, delimiter=",")
    np.savetxt("CUGCtest.csv", dist_UG, delimiter=",")
    np.savetxt("CtrainCtest.csv", dist_train, delimiter=",")
    print("HW2 was done!\nElapsed time is %.2fs" % (time.time() - start))
    sc.stop()


if __name__ == "__main__":
    # Configure Spark
    APP_NAME = "HW2 Spark Application"
    conf = (SparkConf().setAppName(APP_NAME).setMaster("local[*]").set("spark.shuffle.service.enabled", "false").set("spark.dynamicAllocation.enabled", "false").set("spark.io.compression.codec", "snappy").set("spark.rdd.compress", "true").set("spark.eventLog.enabled", "false"))
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    # sqlContext = SQLContext(sc)
    if os.path.basename(os.getcwd()) == "brainD15":
        authors()
        start = time.time()
    else:
        print("You should move your python script into brainD15 folder.\n")
    main(sc)
