# -*- coding = utf-8 -*-
"""
    Spark version of hw2
"""
import os
import sys
import time
import numpy as np
import re
try:
    sc and spark
except NameError as e:
    import findspark
    findspark.init()
# from operator import add
# from functools import reduce
from sklearn.preprocessing import scale
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark import StorageLevel
# from collections import OrderedDict
# from pyspark.sql import SQLContext
# from pyspark.mllib.linalg import Vectors
# from pyspark.mllib.linalg.distributed import RowMatrix
# from pyspark.sql.types import StructType, StructField
# from pyspark.sql.types import FloatType
# from pyspark.sql.functions import log
# from pyspark.mllib.stat import Statistics
# from pyspark.ml.feature import StandardScaler
# from pyspark.sql import DataFrame


def authors():
    print("@authors: Yue Peng, Ludan Zhang, Jiachen Zhang\n")


def stack_(m1, m2):
    return np.concatenate((m1, m2))


def cov_(mat):
    return np.cov(mat.T)


def toCSV(mat):
    return np.savetxt("Ctest.csv", mat, delimiter=",")


def svd_(mat):
    u, s, v = np.linalg.svd(mat, full_matrices=False, compute_uv=True)
    g = np.dot(np.diag(s), v)
    UG = np.dot(u, np.dot(np.diag(s), v))
    return [u, g, UG]


def main(sc, files):

    rdd = sc.wholeTextFiles(",".join(files)).persist(StorageLevel.DISK_ONLY)
    rdd_id = rdd.map(lambda x: (float(x[0][re.search(r"\d{6}", x[0]).span()[0]:re.search(r"\d{6}", x[0]).span()[1]]), x[1])).persist(StorageLevel.DISK_ONLY)
    lines = rdd_id.mapValues(lambda x: x.rstrip().split("\n")).persist(StorageLevel.DISK_ONLY)
    mat = lines.mapValues(lambda x: [*map(str.split, x)]).persist(StorageLevel.DISK_ONLY)
    mat_num = mat.mapValues(lambda x: np.array(x).astype(float)).persist(StorageLevel.DISK_ONLY)
    scaled_mat = mat_num.mapValues(lambda x: scale(x)).persist(StorageLevel.DISK_ONLY)
    train = scaled_mat.filter(lambda x: x[0] <= 211316).persist(StorageLevel.DISK_ONLY)
    test = scaled_mat.filter(lambda x: x[0] > 211316).persist(StorageLevel.DISK_ONLY)

    Xs_train_mat, Xs_test_mat = train.map(lambda x: ("train", x[1])).persist(StorageLevel.DISK_ONLY), test.map(lambda x: ("test", x[1])).persist(StorageLevel.DISK_ONLY)
    XsTrain, XsTest = Xs_train_mat.reduceByKey(stack_).persist(StorageLevel.DISK_ONLY), Xs_test_mat.reduceByKey(stack_).persist(StorageLevel.DISK_ONLY)

    cov_Xs_train, cov_Xs_test = XsTrain.mapValues(cov_).persist(StorageLevel.DISK_ONLY), XsTest.mapValues(cov_).persist(StorageLevel.DISK_ONLY)

    cov_Xs_test.mapValues(toCSV)
    XsTrain_svd = XsTrain.mapValues(svd_).persist(StorageLevel.DISK_ONLY)

   
    cov_UG = np.cov(UG.T)
    dist_UG = np.array(np.linalg.norm(cov_UG - cov_Xs_test, ord="fro")).reshape(1, 1)
    dist_train = np.array(np.linalg.norm(cov_Xs_train - cov_Xs_test, ord="fro")).reshape(1, 1)
    print("The closeness between matrix CUG and Ctest is %f\nThe closeness between matrix Ctrain and Ctest is %f\n" % (dist_UG, dist_UG))
    print("Until here, elapsed time is %.2fs\n" % (time.time() - start))
    np.savetxt("Up5.csv", u, delimiter=",")
    np.savetxt("Gp5.csv", g, delimiter=",")
    np.savetxt("CUGp5.csv", cov_UG, delimiter=",")
    np.savetxt("Ctrainp5.csv", cov_Xs_train, delimiter=",")
    np.savetxt("Ctestp5.csv", cov_Xs_test, delimiter=",")
    np.savetxt("CUGCtestp5.csv", dist_UG, delimiter=",")
    np.savetxt("CtrainCtestp5.csv", dist_train, delimiter=",")
    print("HW2 was done!\nElapsed time is %.2fs" % (time.time() - start))
    sc.stop()


if __name__ == "__main__":
    # Configure Spark
    APP_NAME = "HW2 Spark Application"
    conf = (SparkConf().setAppName(APP_NAME).set("spark.shuffle.service.enabled", "false").set("spark.io.compression.codec", "snappy").set("spark.rdd.compress", "true").set("spark.eventLog.enabled", "false"))
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    # sqlContext = SQLContext(sc)
    if os.path.basename(os.getcwd()) == "brainD300":
        authors()
        start = time.time()
        files = sorted([s for s in os.listdir(os.getcwd()) if s.endswith(".txt")])
        main(sc, files)
    else:
        sys.exit("You should move your python script into brainD15 folder.\n")
    
