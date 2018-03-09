import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.storage.StorageLevel
import java.io.{File, FileWriter, BufferedWriter, PrintWriter} // StringWriter
import breeze.linalg.{DenseMatrix, svd, diag} //DenseVector, convert, cov
import breeze.stats.covmat
import au.com.bytecode.opencsv.CSVWriter
import scala.collection.JavaConverters._

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

  def denseToCSV(mat: DenseMatrix[Double]): java.util.List[Array[String]] = {
    mat.toString.split("\n").map(_.split("\\s+")).toList.asJava
  }

  def saveCSV(list: java.util.List[Array[String]], s:String): Unit = {
    val out = new BufferedWriter(new FileWriter(System.getProperty("user.dir") + "/" + s + ".csv"))
    val writer = new CSVWriter(out)
    writer.writeAll(list)
    writer.close()
  }
  
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setAppName("Spark version of HW2")
    conf.setMaster("local[*]")
    conf.set("spark.memory.offHeap.enabled", "true")
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.memory.fraction", "0.2")
    conf.set("spark.memory.offHeap.size", "2")
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


    val U_csv = U.mapValues(denseToCSV(_))
    saveCSV(U_csv.map(_._2).collect()(0), "Up5")

    val G_csv = G.mapValues(denseToCSV(_))
    saveCSV(G_csv.map(_._2).collect()(0), "Gp5")

    val CUG_csv = CUG.mapValues(denseToCSV(_))
    saveCSV(CUG_csv.map(_._2).collect()(0), "CUGp5")

    val Ctrain_csv = Ctrain.mapValues(denseToCSV(_))
    saveCSV(Ctrain_csv.map(_._2).collect()(0), "Ctrainp5")

    val Ctest_csv = Ctest.mapValues(denseToCSV(_))
    saveCSV(Ctest_csv.map(_._2).collect()(0), "Ctestp5")

    val writerCUGCtest = new PrintWriter(new File("CUGCtestp5.csv"))
    writerCUGCtest.write(CUGCtest.collect()(0)._2.toString + "\n")
    writerCUGCtest.close()

    val writerCtrainCtest = new PrintWriter(new File("CtrainCtestp5.csv"))
    writerCtrainCtest.write(CtrainCtest.collect()(0)._2.toString + "\n")
    writerCtrainCtest.close()
    sc.stop()
  }
}
