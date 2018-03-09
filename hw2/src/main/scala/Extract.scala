import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.storage.StorageLevel
import java.io.{File, PrintWriter}
import breeze.linalg.{DenseMatrix, svd, diag} //DenseVector, convert, cov
import breeze.stats.covmat
//import scala.math
//import breeze.linalg._


object Extract {
  def stringToArray(s: String): Array[String] = s.split(" ")

  def stringIndex(s: String): Tuple2[Int, Int] = (s.indexOfSlice("brainD15") + 9, s.indexOfSlice("brainD15") + 23)

  def colMeans(arr: Array[Array[Double]]): Array[Double] = arr.transpose.map(_.sum)//arr.map(Vector(_)).reduce(_ + _).toArray

  def scaling(arr: Array[Array[Double]]): Array[Array[Double]] = {
    val cm = colMeans(arr)
    val tmp = arr.map(_.zip(cm).map(x => x._1 - x._2))
    val ss = tmp.map(_.map(math.pow(_, 2)))
    val stddev = ss.transpose.map(_.sum).map(_/arr.size).map(math.sqrt(_))
    val mat = arr.map(_.zip(cm).map(x => x._1 - x._2)).map(_.zip(stddev).map(x=>x._1/x._2))
    mat
  }

  def arrayToDenseMatrix(arr: Array[Array[Double]]): DenseMatrix[Double] = {
    val mat = DenseMatrix(arr.flatMap(_.toList).toList).reshape(arr(0).length, arr.size).t
    mat
  }

  def runSVD(mat: DenseMatrix[Double]): Tuple3[DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double]] = {
    val U = svd(mat).U
    val S = diag(svd(mat).S)
    val Vt = svd(mat).Vt(0 to mat.rows - 1, ::)
    val G = S * Vt
    val UG = U * G
    val res = (U, G, UG)
    res
  }

  def runCOV(mat: DenseMatrix[Double]): DenseMatrix[Double] = covmat(mat)

  def normFrobenius(mat: DenseMatrix[Double]): Double = {
    math.sqrt(mat.mapValues(x => math.pow(x, 2)).toArray.reduce(_ + _))
  }

  def denseToCsv(mat: DenseMatrix[Double]): Array[String] = {
    mat.toString.split("\n").map(_.split("\\s+")).map(_.mkString(","))
  }


  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setAppName("Spark version of HW2")
    conf.setMaster("local[*]")
    conf.set("spark.memory.offHeap.enabled", "true")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    val files_in_dir = new File(System.getProperty("user.dir"))
    val files = files_in_dir.listFiles.map(_.toString)
    val files_txt = files.map(x => x.slice(stringIndex(x)._1, stringIndex(x)._2)).filter(_.startsWith("sub")).sorted
    val files_path = files_txt.mkString(",")
    val input = sc.wholeTextFiles(files_path)//.persist(StorageLevel.DISK_ONLY)
    val input2 = input.mapValues(x => x.split("\n"))//.persist(StorageLevel.DISK_ONLY)
    val input3 = input2.mapValues(x => x.map(x => stringToArray(x)))//.persist(StorageLevel.DISK_ONLY)
    val mat = input3.mapValues(x => x.map(_.map(_.toDouble)))//.persist(StorageLevel.DISK_ONLY)
    val id_pattern = "\\d{6}".r
    val mat_Double = mat.map(x => (id_pattern.findAllIn(x._1).toSeq.apply(0).toDouble, x._2))//.persist(StorageLevel.DISK_ONLY)
    val mat_scaled = mat_Double.mapValues(scaling(_))//.persist(StorageLevel.DISK_ONLY)
    val mat_train = mat_scaled.filter(_._1 <= 211316).map(x => ("train", x._2))//.persist(StorageLevel.DISK_ONLY)
    val mat_test = mat_scaled.filter(_._1 > 211316).map(x => ("test", x._2))//.persist(StorageLevel.DISK_ONLY)
    val XsTrain = mat_train.reduceByKey(_ ++ _)//.persist(StorageLevel.DISK_ONLY)
    val XsTest = mat_test.reduceByKey(_ ++ _)//.persist(StorageLevel.DISK_ONLY)
    val trains = XsTrain.mapValues(arrayToDenseMatrix(_))
    val tests = XsTest.mapValues(arrayToDenseMatrix(_))
    val Ctrain = trains.mapValues(runCOV(_))
    val Ctest = tests.mapValues(runCOV(_))
    val svdTuple = trains.mapValues(runSVD(_))
    val U = svdTuple.mapValues(_._1)
    val G = svdTuple.mapValues(_._2)
    val CUG = svdTuple.mapValues(x => runCOV(x._3))
    val UG_and_test = CUG ++ Ctest
    val UGTest = UG_and_test.map(x => ("UGTest", x._2))
    val CUG_Ctest = UGTest.reduceByKey(_ - _)
    val train_and_test = Ctrain ++ Ctest
    val TrainTest = train_and_test.map(x => ("TrainTest", x._2))
    val Ctrain_Ctest = TrainTest.reduceByKey(_ - _)
    val CUGCtest = CUG_Ctest.mapValues(x => normFrobenius(x))
    val CtrainCtest = Ctrain_Ctest.mapValues(x => normFrobenius(x))

    val writerU = new PrintWriter(new File("U.csv"))
    val U_csv = U.mapValues(denseToCsv(_))
    U_csv.mapValues(_.foreach(x => writerU.write(x + "\n")))
    writerU.close()

    val writerG = new PrintWriter(new File("G.csv"))
    val G_csv = G.mapValues(denseToCsv(_))
    G_csv.mapValues(_.foreach(x => writerG.write(x + "\n")))
    writerG.close()

    val writerCUG = new PrintWriter(new File("CUG.csv"))
    val CUG_csv = CUG.mapValues(denseToCsv(_))
    CUG_csv.mapValues(_.foreach(x => writerCUG.write(x + "\n")))
    writerCUG.close()

    val writerCtrain = new PrintWriter(new File("Ctrain.csv"))
    val Ctrain_csv = Ctrain.mapValues(denseToCsv(_))
    Ctrain_csv.mapValues(_.foreach(x => writerCtrain.write(x + "\n")))
    writerCtrain.close()

    val writerCtest = new PrintWriter(new File("Ctest.csv"))
    val Ctest_csv = Ctest.mapValues(denseToCsv(_))
    Ctest_csv.mapValues(_.foreach(x => writerCtest.write(x + "\n")))
    writerCtest.close()

    val writerCUGCtest = new PrintWriter(new File("CUGCtest.csv"))
    CUGCtest.mapValues(x => writerU.write(x.toString + "\n"))
    writerCUGCtest.close()

    val writerCtrainCtest = new PrintWriter(new File("CtrainCtest.csv"))
    CtrainCtest.mapValues(x => writerU.write(x.toString + "\n"))
    writerCtrainCtest.close()

  }
}
