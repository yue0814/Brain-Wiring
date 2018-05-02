import org.apache.spark.{SparkConf, SparkContext}
import scala.util.matching.Regex
import java.io.File


object Extract {
  def stringToArray(s: String): Array[String] = s.split(" ")

  def main(args: Array[String]): Unit = {
    val files_in_dir = new File(System.getProperty("user.dir"))
    val files = files_in_dir.listFiles.map(_.toString)
    val files_path = files.mkString(",")
    val conf = new SparkConf()
    conf.setAppName("Spark version of HW2")
    conf.setMaster("local[*]")
    val sc = new SparkContext(conf)
    val input = sc.wholeTextFiles(files_path)
    val input2 = input.mapValues(x => x.split("\n"))
    val input3 = input2.mapValues(x => x.map(x => stringToArray(x)))
    val mat = input3.mapValues(x => x.map(_.map(_.toFloat)))
    val id_pattern = raw"\d{6}".r
    val mat_float = mat.map(x => (id_pattern.findAllIn(x._1).toList(0).toFloat, x._2))
  }
}
