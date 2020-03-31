import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import com.johnsnowlabs.nlp.annotator.BertEmbeddings
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach
import com.johnsnowlabs.nlp.training.CoNLL
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession

object NerModelCreator {


  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("NER Model Creator")
      .config("spark.master", "local")
      .getOrCreate()

    val trainingConll = CoNLL(conllLabelIndex = 2).readDataset(spark, "src/main/resources/trivialTrainingData.txt")

    val bert = BertEmbeddings.pretrained("bert_base_uncased", "en")
      .setInputCols("sentence", "token")
      .setOutputCol("bert")
      .setCaseSensitive(false)
      .setPoolingLayer(0)

    val nerTagger = new NerDLApproach()
      .setInputCols("sentence", "token", "bert")
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMaxEpochs(5)
      .setRandomSeed(0)
      .setVerbose(4)
      .setValidationSplit(0.2f)

    val pipeline = new Pipeline().
      setStages(Array(
        bert,
        nerTagger
      ))

    pipeline
      .fit(trainingConll)
      .write.save(s"target/trainedModel-${LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYY-MM-dd_HHmmss"))}")
  }
}
