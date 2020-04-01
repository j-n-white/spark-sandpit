import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{BertEmbeddings, NerDLModel, SentenceDetector, Tokenizer}
import com.johnsnowlabs.nlp.annotators.ner.NerConverter
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object NamedEntityRecognition {

  def addKeyword(output: Seq[String], currentKeyword: String): Seq[String] =
    if(currentKeyword != null) output :+ currentKeyword else output

  val getKeywords: (Seq[String], Seq[String]) => Seq[String] = (tokens, tags) => {
    var output = Seq.empty[String]
    var currentKeyword: String = null
    for (i <- tokens.indices) {
      if ("B-KEYWORD".equals(tags(i))) {
        output = addKeyword(output, currentKeyword)
        currentKeyword = tokens(i)
      } else if ("I-KEYWORD".equals(tags(i))) {
        if (i == 0 || "O".equals(tags(i-1))) {
          output = addKeyword(output, currentKeyword)
          currentKeyword = tokens(i)
        } else {
          currentKeyword = currentKeyword + tokens(i)
        }
      }
    }
    addKeyword(output, currentKeyword)
  }

  def main(args: Array[String]): Unit = {
    val profileFileName = "src/main/resources/blog/2020-03-30-Finding-the-right-words.md"
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

    val loaded_ner_model = NerDLModel.load("target/trainedModel-2020-03-31_151441/stages/1_NerDLModel_83e23a7253bc")
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


    val getKeywordsUdf = udf(getKeywords)
    resultsFrame
      .withColumn("Keywords", getKeywordsUdf($"token.result", $"ner.result"))
      .withColumn("explodedKeywords", explode($"Keywords"))
      .select("explodedKeywords")
      .groupBy("explodedKeywords")
      .agg(count("*").as("count"))
      .orderBy($"count".desc)
      .repartition(1)
      .write.csv(s"target/keywordCount-${LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYY-MM-dd_HHmmss"))}")
  }
}
