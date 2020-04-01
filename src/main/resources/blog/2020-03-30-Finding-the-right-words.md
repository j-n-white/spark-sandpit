---
published: true
author: gstulgyte
layout: default_post
category: Tech
title: 'Finding the right words: a comparison of Named Entity Recognition tools'
tags: >-
  Machine Learning, Natural Language Processing, Named Entity Recognition,
  Cloud, Tech
summary: >-
  We look at the most popular Natural Language Processing (NLP) libraries out there by creating a simple Named Entity Recognition (NER) task. Then we explore the results and discuss how the open-source libraries fare against each other and cloud-based NLP solutions.
contributors:
  - sshiells
  - jwhite
---
Natural Language Processing (NLP), which encapsulates a large number of areas which can be used to aid interactions between computers and human languages, has been around for quite some time. With such a vast landscape of libraries to choose from, it can be difficult to know where to start. 

To get a better idea of how different tools fare against each other, we looked into comparing different NLP libraries by looking at named entity recognition (NER) - locating and classifying named entities in a body of text. As our dataset, we used Scott Logic’s internal consultant profiles, which are a bit like CVs that we present to our clients, and wanted to see if we could use NER to identify which of Scott Logic’s consultants had experience with what technologies.

## Getting Started

We wanted to look at the most popular cloud-based NLP services and open source libraries out there.

Before starting, we came up with a plan on how we would test each library and how we could compare the results. We realised that we had to hand over our consultant profiles from across our business so the first step was anonymizing the data. Then we created a dummy consultant profile and would run this through our trained models, then measure how accurately each of our models identified the technologies mentioned in the profile. Using a dummy profile allowed us to get a bit creative and to include older technologies (e.g. Fortran) that were not present in any of the training data to see if the models would pick any of these up.

To determine the effectiveness of each library we devised the following criteria to help us evaluate the results.

- Full and complete matches count as a success.
- Partial matches count as failures. For example, if the string contains “.NET Core” and the library matches “NET Core” but not the “.” then this would be considered as a failure.
- If the library matches a version it is still a success as long as it is part of the same entity e.g. matching [ABC] 4.5, 5.5 [ABC 4.5], 5.5 or [ABC 4.5, 5.5] is a pass but [ABC 4.5], [5.5] is a pass and a false positive.
- Anything that is tagged as an entity that is not a technology is counted as a false positive.

For each library, we wanted to see how it coped with different amounts of training data. For this data set, we decided that we would run each library against, 0, 1, 2, 5, 10, 15, 20 and 25 profiles worth of training data. A consultant profile can contain anywhere between 20-70 technology mentions, often with multiple mentions on a single line.

It is worth noting that this is a much smaller dataset than would normally be used for NLP. The main reason we decided to restrict the dataset to the number of profiles is that we were dealing with a fairly limited dataset and we started to see encouraging levels of accuracy after only a few profiles.

## The Cloud

While there are dozens of cloud-based NLP solutions out there, we decided to focus on the heavyweights: Amazon Web Services (AWS), Microsoft Azure, and Google Cloud. However, it quickly became clear that Azure is not suitable for this experiment as it does not currently support custom entity recognition. We were left with two tools to explore - **AWS Comprehend** and **Google Cloud AutoML**. Overall, we found that both of the cloud providers were quite easy to get started with and there was no programming knowledge strictly required, but there were some note-worthy differences between the two, namely:

- The data set size limits.
- The annotating and training process.
- Training costs and time.

### AWS Comprehend
We relied on [this blog post](https://aws.amazon.com/blogs/machine-learning/build-a-custom-entity-recognizer-using-amazon-comprehend/) on the AWS Machine Learning Blog. AWS Comprehend lets you train a custom recognizer in two ways - with training documents or an entity list, or with training documents and an annotations list. Both of these have different advantages and led to slightly different results in the end. 

**Pros:**

- Easy to set up - using an entity list is the quickest way of getting started with NLP as AWS generates the annotations itself.
- The custom model can be accessed through the AWS Console or exposed through AWS CLI.
- Affordable - training a model on AWS Comprehend took 20-17 minutes and we spent roughly $6-7 in total.

**Cons:**

- When accessed through AWS console, the model returns a zipped tarball with JSON data - it is not immediately human-readable.
- The **entity list** option is unsuitable for data sets with more complex entities as there is no way to review the training data.
- Using an **annotations list** in an AWS-specific format is more accurate but also more hands-on and time-consuming.
- At least 1000 documents (lines with an entity occurrence) are required to train a custom recognizer, i.e. at least 25 consultant profiles.  As a consequence, comparing AWS Comprehend’s performance to the remaining libraries was a challenge, although it is worth noting that NLP is generally more suitable for larger data sets.

### Google Cloud AutoML
Google Cloud AutoML also gives you two ways of annotating the training data - you can either annotate the training documents prior to uploading them and review the annotations through Google Cloud’s visual annotation tool, AutoML Natural Language UI, or you can fully annotate the documents through AutoML Natural Language UI. We followed [this quickstart](https://cloud.google.com/natural-language/automl/docs/quickstart) on Google Cloud.

**Pros:**

- Easy to use - AutoML Natural Language UI is very handy.
- Google Cloud provides their own Python helper tool for annotating, converting, and uploading the documents to Google Cloud.
- Flexible data set size limits - Cloud AutoML requires anywhere between 50-100,000 documents for model training. 
- Like with AWS Comprehend, the custom model can be queried through Google Cloud UI or accessed through REST API / Python.
- Using Google Cloud UI immediately shows you results.

**Cons:**

- Cost and time. Training a model on Google Cloud can take several hours and we would have been billed roughly $40 if not the free trial. This is not necessarily reflective of the real training cost as some of the initial training attempts failed due to incorrect configuration or insufficient training documents. 

## Open Source Libraries
As well as all of the major cloud providers offering some NLP capabilities there are a large number of open-source libraries available. For our experiment, we picked a handful of some of the most popular of these. All of the libraries we chose offered the flexibility of being able to use pre-trained models as well as a means of training our own model. Due to the nature of our dataset, the pre-trained models were not useful for identifying the different technologies contained within our consultant profiles so we opted to train our own. Different open-source libraries were variable in how much they could be configured. We only did limited experimentation on different configurations to get up and running quickly with something that produced reasonable results. If you are using a specific library then it would be worth spending some more time evaluating how the different configurations work for your use case.

### spaCy
spaCy is a free, open-source library for NLP in Python. spaCy offers much more than the NER capabilities, it also offers part-of-speech tagging, labeled dependency parsing, syntax-driven sentence segmentation and much more.

**Pros:**

- Simple to use.
- Interactive online tutorial: [https://course.spacy.io/](https://course.spacy.io/)
- Ability to run with little or no training data.
- Yielded good results quickly.

**Cons:**

- The annotation notation format is relatively complex.
- Provided more false positives than the other libraries we tested.

### Apache OpenNLP
Apache OpenNLP is a machine learning toolkit that supports most of the common NLP tasks. Apache OpenNLP is written in java and has been in development for over 15 years.

**Pros:**

- Good documentation with a consistent structure to API.
- There is a sizable set of pre-trained models ([Available here](http://opennlp.sourceforge.net/models-1.5/)).
- Simple XML like annotation format.

**Cons:**

- Slightly fewer correct matches than other libraries.

### Epic
Epic is a high-performance statistical parser written in Scala which provides a framework for building complex structured prediction models. 

**Pros:**

- Quick to get started following this [blog post](https://towardsdatascience.com/simple-nlp-search-in-your-application-step-by-step-guide-in-scala-22ca1ce3e475).

**Cons:**

- Whilst we were working on this investigation the library was archived and is no longer maintained.
- Limited pre-trained models.
- Need to write own function to parse training data.

### Spark NLP
Spark NLP is a library written for python, java, and Scala. It is written on top of Apache Spark and Tensorflow.

**Pros:**

- Easy integration with Spark for additional processing.
- Easy to run pre-trained models.
- Lots of configuration options.
- Comparable accuracy to cloud based solutions.

**Cons:**

- Difficult to set up named entity recognition. We relied on this [blog post](https://towardsdatascience.com/named-entity-recognition-ner-with-bert-in-spark-nlp-874df20d1d77)).
- Configuration options are confusing. Our first attempt resulted in a configuration which wasn’t actually using any of the training data to train the model and hence was extremely inaccurate.
- We also ran into an issue when running locally was repeated running of the code resulted in large quantities of data being produced in a temp directory that filled the hard drive.

### A Common Theme
One of the things shared amongst almost all of the libraries that we tested was that the annotation of training data was not straightforward, very time consuming and also very repetitive. Each library also had their own way of annotating the training data. For most of the libraries, we ended up writing some custom scripts to automatically annotate the training data in the desired format as a first pass, and then manually verifying that the annotations were correct. This ended up saving us a lot of time and allowed us to get to processing our tests more quickly. Both of the cloud-based services offered some ways of getting around the manual annotating process, which was a clear advantage - Google Cloud AutoML, for example, had a pre-written helper script as well as a visual, user-friendly annotation tool that made it much easier to ensure the accuracy of the annotations that we were applying to the training data. 

## The Results

Here are the results for how many entities were correctly matched with different amounts of training data.

[![Graph showing the number of correct matches against the number of profiles used as training data. The expected number of matches was 56. All the libraries showed rapid improvements of accuracy up to 5 profiles but beyond that the increase plateaued in the region of 80% - 90% correct matches with 25 profiles.]({{site.baseurl}}/gstulgyte/assets/CorrectMatches.PNG)]({{site.baseurl}}/gstulgyte/assets/CorrectMatches.PNG)

Overall we were impressed with how quickly we got to a high level of accuracy, with all of the libraries we chose performing relatively well. Interestingly, while none of the libraries performed well when the number of used profiles was less than 5, increasing the number of profiles used for training did not always lead to significant improvement. In spaCy’s case, the number of correct matches dropped when more than 10 profiles were used for training (at 25 training profiles, the number of correct matches went down by 5 in comparison to 10 training profiles). 

Here are the results for the entities incorrectly matched i.e. items identified that were not technologies or partial matches.

[![Graph showing incorrect matches against number of profiles used as training data. All the libraries spike at low numbers of profiles and gradualy decline until they have 0-2 incorrect matches with 25 profiles. The exception to this is spaCy which actually has an increase in incorrect matches with 25 profiles.]({{site.baseurl}}/gstulgyte/assets/IncorrectMatchesTopLegend.PNG)]({{site.baseurl}}/gstulgyte/assets/IncorrectMatchesTopLegend.PNG)

Similar to the correct matches the results generally get better with more training data but this is not always the case. For example spaCy had a spike in incorrect matches when 25 profiles were used.

The next graph shows the results when we used the most training data.

[![Graph showing correct and incorrect matches for the libraries for 25 profiles. The data shown is: Open NLP: 44 correct 2 incorrect, Epic: 46 correct 1 incorrect, spaCy: 47 correct 8 incorrect, Spark NLP: 51 correct 2 incorrect, Google CLoud AutoML: 52 correct 0 incorrect, AWS Comprehend (entity list): 52 correct 0 incorrect, AWS Comprehend: 53 correct 2 incorrect.]({{site.baseurl}}/gstulgyte/assets/CorrectPerLibrary.PNG)]({{site.baseurl}}/gstulgyte/assets/CorrectPerLibrary.PNG)

In the profile we were analysing there were a total of 56 mentions of technologies. All of these libraries managed to identify most of them. The worst was Open NLP which identified 44 of them (78.57%) and the best was AWS Comprehend which identified 53 (93.00%). All of the libraries had very few incorrect matches, the highest was spaCy with 8.

## Conclusion
We can see from the results that all the libraries tested performed well in this situation identifying correctly most of the technologies.

While both AWS Comprehend and GCP Cloud AutoML are paid services, the cloud-based libraries gave slightly better results and, on average, less incorrect matches. Therefore, the cloud-based NLP tools are definitely worth giving a go, although cloud may not be the best choice if complete accuracy is not your main priority or if the cost is an issue (however, as mentioned earlier, the cost depends on the service). Despite the associated cost, cloud-based language processors are likely to be better equipped to handle large volumes of data due to the sheer computing power required to train a custom model. 

Spark NLP comes out on top of the non-cloud solutions with comparable numbers of correct and incorrect matches.

We mentioned earlier that we included Fortran in our test profile to see how well these libraries could identify an entity that was not in the training data. Most of the libraries correctly identified this entity when used in a sentence however only the cloud solutions managed to correctly identify it in the skills list. This is understandable as for most of these libraries we had to process sentences individually so in the skills list Fortran was just a single word with no other context.

However it is worth noting that our data is based upon how well it identifies entities within one profile. To get a more accurate evaluation we would need to repeat this with multiple varied profiles.
