import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object TrainingDataCreator {

  val createTags: (Seq[String], Seq[String]) => Seq[String] = (tokens, techTags) => {
    var tags = Seq.fill(tokens.length)("O")
    var index = 0;
    while (index < tokens.length && tokens.indexOfSlice(techTags, index) != -1) {
      val start = tokens.indexOfSlice(techTags, index)
      for (i <- start until start + techTags.length) {
        val tag = if (i == start) "B-TECH" else "I-TECH"
        tags = tags.updated(i, tag)
      }
      index = start + techTags.length
    }
    tags
  }

  val createOutputLines: (Seq[String], Seq[String], Seq[String]) => String = (tokens, pos, tags) => {
    var output = Seq.empty[String]
    for (i <- tokens.indices) {
      output = output :+ tokens(i) + " " + pos(i) + " " + tags(i) + "\n"
    }
    output.mkString("")
  }

  val combineTags: (Seq[Seq[String]]) => Seq[String] = (tegSeqs) => tegSeqs.reduce((x, y) => {
    var mergedTags = x
    for (i <- x.indices) {
      if (x(i).equals("O")) {
        mergedTags = mergedTags.updated(i, y(i))
      }
    }
    mergedTags
  })

  def main(args: Array[String]): Unit = {
    val profileFileName = "src/main/resources/Profile1.txt"
    val techFileName = "src/main/resources/Tech.csv"
    val spark = SparkSession.builder
      .appName("Training Data Creator")
      .config("spark.master", "local")
      .getOrCreate()

    import spark.implicits._

    val tech = spark.read
      .textFile(techFileName)
      .withColumnRenamed("value", "text")

    val profile = spark.read
      .textFile(profileFileName)
      .withColumnRenamed("value", "text")
    val pipeline = PretrainedPipeline("explain_document_dl", lang = "en")

    val tagUdf = udf(createTags)
    val combineTagsUdf = udf(combineTags)
    val outputLinesUdf = udf(createOutputLines)

    val techTags = pipeline.transform(tech).withColumn("techTags", $"token.result").select("techTags")

    pipeline.transform(profile)
      .crossJoin(techTags)
      .withColumn("tokens", $"token.result")
      .withColumn("posResult", $"pos.result")
      .withColumn("tags", tagUdf($"tokens", $"techTags"))
      .select("tokens", "posResult", "tags", "techTags")
      .groupBy("tokens")
      .agg(
        first("posResult").alias("posResult"),
        combineTagsUdf(collect_list($"tags")).alias("tags")
      )
      .withColumn("outputLines", outputLinesUdf($"tokens", $"posResult", $"tags"))
      .select("outputLines")
      .repartition(1)
      .write.text(s"target/trainingData${LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYY-MM-dd_HHmmss"))}")
  }

}
