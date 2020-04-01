import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object TrainingDataCreator {

  val createTags: Seq[String] => Seq[String] = tokens =>  Seq.fill(tokens.length)("O")

  val createOutputLines: (Seq[String], Seq[String], Seq[String]) => String = (tokens, pos, tags) => {
    var output = Seq.empty[String]
    for (i <- tokens.indices) {
      output = output :+ tokens(i) + " " + pos(i) + " " + tags(i) + "\n"
    }
    output.mkString("")
  }

  def main(args: Array[String]): Unit = {
    val blogFileName = "src/main/resources/blog/2020-03-19-offscreen-canvas.md"
    val spark = SparkSession.builder
      .appName("Training Data Creator")
      .config("spark.master", "local")
      .getOrCreate()

    import spark.implicits._

    val profile = spark.read
      .textFile(blogFileName)
      .withColumnRenamed("value", "text")
    val pipeline = PretrainedPipeline("explain_document_dl", lang = "en")

    val tagUdf = udf(createTags)
    val outputLinesUdf = udf(createOutputLines)


    pipeline.transform(profile)
      .withColumn("tokens", $"token.result")
      .withColumn("posResult", $"pos.result")
      .withColumn("tags", tagUdf($"tokens"))
      .withColumn("outputLines", outputLinesUdf($"tokens", $"posResult", $"tags"))
      .select("outputLines")
      .repartition(1)
      .write.text(s"target/trainingData-${LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYY-MM-dd_HHmmss"))}")
  }

}
