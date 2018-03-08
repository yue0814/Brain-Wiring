import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.storage.StorageLevel
import java.io.File
//import scala.math
//import breeze.linalg._


object Extract {
  def stringToArray(s: String): Array[String] = s.split(" ")

  def colMeans(arr: Array[Array[Double]]): Array[Double] = arr.transpose.map(_.sum)//arr.map(Vector(_)).reduce(_ + _).toArray

  def scaling(arr: Array[Array[Double]]): Array[Array[Double]] = {
    val cm = colMeans(arr)
    val tmp = arr.map(_.zip(cm).map(x => x._1 - x._2))
    val ss = tmp.map(_.map(math.pow(_, 2)))
    val stddev = ss.transpose.map(_.sum).map(_/arr.size).map(math.sqrt(_))
    val mat = arr.map(_.zip(cm).map(x => x._1 - x._2)).map(_.zip(stddev).map(x=>x._1/x._2))
    mat
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
    val files_txt = files.map(_.slice(34, 47)).filter(_.startsWith("sub")).sorted
    val files_path = files_txt.mkString(",")
    val input = sc.wholeTextFiles(files_path).persist(StorageLevel.DISK_ONLY)
    val input2 = input.mapValues(x => x.split("\n")).persist(StorageLevel.DISK_ONLY)
    val input3 = input2.mapValues(x => x.map(x => stringToArray(x))).persist(StorageLevel.DISK_ONLY)
    val mat = input3.mapValues(x => x.map(_.map(_.toDouble))).persist(StorageLevel.DISK_ONLY)
    val id_pattern = "\\d{6}".r
    val mat_Double = mat.map(x => (id_pattern.findAllIn(x._1).toSeq.apply(0).toDouble, x._2)).persist(StorageLevel.DISK_ONLY)
    val mat_scaled = mat_Double.mapValues(scaling(_)).persist(StorageLevel.DISK_ONLY)
    val mat_train = mat_scaled.filter(_._1 <= 211316).map(x => ("train", x._2)).persist(StorageLevel.DISK_ONLY)
    val mat_test = mat_scaled.filter(_._1 > 211316).map(x => ("test", x._2)).persist(StorageLevel.DISK_ONLY)
    val XsTrain = mat_train.reduceByKey(_ ++ _).persist(StorageLevel.DISK_ONLY)
    val XsTest = mat_test.reduceByKey(_ ++ _).persist(StorageLevel.DISK_ONLY)
    
  }
}
