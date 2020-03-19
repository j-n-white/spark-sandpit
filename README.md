# Spark Sandpit

## Create training data

1. Add a profile to `/src/main/resources`
1. Change `profileFileName` in `TrainingDataCreator.scala` line 43 to match the profile you added
1. Run `TrainingDataCreator.scala` this will output the data to `target/trainingData-YYYY-MM-dd_HHmmss`
1. Find the text file with the training data in it.
1. Verify it has correctly tagged the data.
1. Copy tech to your training data file.

## Create an Named Entity Recognition model

1. Make sure `TrainingDataCreator.scala` line 19 has the correct path to your training data created from the above steps.
1. Run `TrainingDataCreator.scala` this will output the model data to `target/trainedModel-YYYY-MM-dd_HHmmss`

## Run Named Entity Recognition

1. Change `NamedEntityRecognition.scala` line 40 to the path to the last stage of the output from the previous step e.g. `target/trainedModel2020-03-19_100323/stages/1_NerDLModel_57a1eaf3f640`
1. Run `NamedEntityRecognition.scala`
1. TODO sort out the output into a readable format.