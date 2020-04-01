# Spark Sandpit

## Create training data

1. Add a blog to `/src/main/resources/blog`
1. Change `blogFileName` in `TrainingDataCreator.scala` line 21 to match the blog you added
1. Run `TrainingDataCreator.scala` this will output the data to `target/trainingData-YYYY-MM-dd_HHmmss`
1. Find the text file with the training data in it.
1. Manually tag the keywords
    1. In the training data each line contains 3 parts. The word, the part of speech and the tag defaulted to `O`.
    1. For single keywords change the `O` to `B-KEYWORD`.
    1. For multiple keywords (e.g. Google Chrome) change the `O` for the first word to `B-KEYWORD` and change all subsequent ones to `I-KEYWORD`. 
1. Copy tech to your training data file.

## Create an Named Entity Recognition model

1. Make sure `TrainingDataCreator.scala` line 19 has the correct path to your training data created from the above steps.
1. Run `TrainingDataCreator.scala` this will output the model data to `target/trainedModel-YYYY-MM-dd_HHmmss`

## Run Named Entity Recognition

1. Change `NamedEntityRecognition.scala` line 36 to the path to the last stage of the output from the previous step e.g. `target/trainedModel2020-03-19_100323/stages/1_NerDLModel_57a1eaf3f640`
1. Run `NamedEntityRecognition.scala`
1. This will out put a csv of keywords and counts to `target/keywordCount-YYYY-MM-dd_HHmmss`