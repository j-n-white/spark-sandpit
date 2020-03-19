import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{BertEmbeddings, NerDLModel, SentenceDetector, Tokenizer}
import com.johnsnowlabs.nlp.annotators.ner.NerConverter
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession

object NamedEntityRecognition {


  def main(args: Array[String]): Unit = {
    val profileFileName = "src/main/resources/exampleProfile.txt"
    val spark = SparkSession.builder
      .appName("Named Entity Recognition")
      .config("spark.master", "local")
      .getOrCreate()

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

    val loaded_ner_model = NerDLModel.load("target/trainedModel2020-03-19_100323/stages/1_NerDLModel_57a1eaf3f640")
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

    resultsFrame
      .select("token.result", "ner.result")
      .show(truncate = false)
  }
}
