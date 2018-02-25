# -*- coding = utf-8 -*-
"""
    Spark version of hw2
"""
import os
import sys
import time
import numpy as np
# from operator import add
from functools import reduce
from sklearn.preprocessing import scale
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark import StorageLevel
# from collections import OrderedDict
# from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
# from pyspark.mllib.linalg.distributed import RowMatrix
# from pyspark.sql.types import StructType, StructField
# from pyspark.sql.types import FloatType
# from pyspark.sql.functions import log
from pyspark.mllib.stat import Statistics
from pyspark.ml.feature import StandardScaler
from pyspark.sql import DataFrame
# findspark.init()


def authors():
    print("@authors: Yue Peng, Ludan Zhang, Jiachen Zhang\n")


def main(sc, files):

    rdd = sc.wholeTextFiles(",".join(files)).persist(StorageLevel.MEMORY_ONLY_SER)
    rdd = rdd.map(lambda x: (x[0][34:], x[1])).persist(StorageLevel.MEMORY_ONLY_SER)
    lines = rdd.mapValues(lambda x: x.rstrip().split("\n")).persist(StorageLevel.MEMORY_ONLY_SER)
    mat = lines.mapValues(lambda x: [*map(str.split, x)]).persist(StorageLevel.MEMORY_ONLY_SER)
    mat_num = mat.mapValues(lambda x: np.array(x).astype(float)).persist(StorageLevel.MEMORY_ONLY_SER)
    scaled_mat = mat_num.mapValues(lambda x: scale(x)).persist(StorageLevel.MEMORY_ONLY_SER)
    scaled_mat_id = scaled_mat.map(lambda x: (float(x[0][3:9]), x[1])).persist(StorageLevel.MEMORY_ONLY_SER)
    scaled_mat_labeled = scaled_mat_id.map(lambda x: (("train" if x[0] <= 211316. else "test"), x[1])).persist(StorageLevel.MEMORY_ONLY_SER)
    



    Xs_train = scaled_mat_id.filter(lambda x: x[0] <= 211316.)
    Xs_test = scaled_mat_id.filter(lambda x: x[0] > 211316.)
    Xs_train_mat, Xs_test_mat = Xs_train.map(lambda x: ("train", x[1])), Xs_test.map(lambda x: ("test", x[1]))
    XsTrain, XsTest = Xs_train_mat.reduce(lambda x, y: np.vstack((x, y))), Xs_test_mat.reduce(lambda x, y: np.vstack((x, y)))

    df = scaled_mat.toDF()
    df.createGlobalTempView("brainD")


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
    conf = (SparkConf().setAppName(APP_NAME).set("spark.shuffle.service.enabled", "false").set("spark.io.compression.codec", "snappy").set("spark.rdd.compress", "true").set("spark.eventLog.enabled", "false"))
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    # sqlContext = SQLContext(sc)
    if os.path.basename(os.getcwd()) == "brainD15":
        authors()
        start = time.time()
    else:
        sys.exit("You should move your python script into brainD15 folder.\n")
    files = sorted([s for s in os.listdir(os.getcwd()) if s.endswith(".txt")])
    main(sc, files)
