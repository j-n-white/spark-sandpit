import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{BertEmbeddings, NerDLModel, SentenceDetector, Tokenizer}
import com.johnsnowlabs.nlp.annotators.ner.NerConverter
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object NamedEntityRecognition {

  val annotateResult: (Seq[String], Seq[String]) => String = (tokens, tags) => {
    var output = Seq.empty[String]
    for (i <- tokens.indices) {
      if ("B-TECH".equals(tags(i))) {
        var annotatedToken = "[TECH: " + tokens(i)
        if (i == tokens.length - 1 || !"I-TECH".equals(tags(i+1))) {
          annotatedToken = annotatedToken + "]"
        }
        output = output :+ annotatedToken
      } else if ("I-TECH".equals(tags(i))) {
        var annotatedToken = tokens(i)
        if (i == 0 || "O".equals(tags(i-1))) {
          annotatedToken = "[TECH: " + annotatedToken
        }
        if (i == tokens.length - 1 || !"I-TECH".equals(tags(i+1))) {
          annotatedToken = annotatedToken + "]"
        }
        output = output :+ annotatedToken
      } else {
        output = output :+ tokens(i)
      }
    }
    output.mkString(" ")
  }

  def main(args: Array[String]): Unit = {
    val profileFileName = "src/main/resources/exampleProfile.txt"
    val spark = SparkSession.builder
      .appName("Named Entity Recognition")
      .config("spark.master", "local")
      .getOrCreate()

    import spark.implicits._

    val profile = spark.read
      .textFile(profileFileName)
      .withColumnRenamed("value", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
      .setCleanupMode("shrink")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val regexTokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val bert = BertEmbeddings.pretrained("bert_base_uncased", "en")
      .setInputCols("sentence", "token")
      .setOutputCol("bert")
      .setCaseSensitive(false)
      .setPoolingLayer(0)

    val loaded_ner_model = NerDLModel.load("target/trainedModel-2020-03-25_163117/stages/1_NerDLModel_0b9b06b77537")
      .setInputCols("sentence", "token", "bert")
      .setOutputCol("ner")

    val converter = new NerConverter()
      .setInputCols("document", "token", "ner")
      .setOutputCol("ner_span")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        regexTokenizer,
        bert,
        loaded_ner_model,
        converter
      ))

    val resultsFrame = pipeline
      .fit(profile)
      .transform(profile)
      .persist()


    val annotateResultUdf = udf(annotateResult)
    resultsFrame
      .withColumn("AnnotatedResult", annotateResultUdf($"token.result", $"ner.result"))
      .select("AnnotatedResult").repartition(1)
      .write.text(s"target/annotatedResult-${LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYY-MM-dd_HHmmss"))}")
  }
}
