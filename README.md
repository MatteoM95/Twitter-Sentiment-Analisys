# Datascience
This project is the final project for the exam `Data Science Lab: process and methods (2021/22)`. The overall score achieved in the final competition was 0.853, which outperformed the given baseline 0.750.

# ABSTRACT
In this report we introduce a possible approach to predict the sentiment associated with a tweet, based on the content of its text. The proposed solution infers the tweets’ sentiment by means of a classification model, that gives overall satisfactory results. In this paper the classification technique used
are logistic regression and linearSVC.

# PROBLEM OVERVIEW
The proposed project is a classification problem on a dataset of tweets. For each tweet the following information is provided:
- *Id*: a numerical identifier of the tweet.
- *date*: the date in which the tweet was published.
- *flag*: the query used to collect the tweet.
- *user*: username of the person who posted the tweet.
- *text*: the text of the tweet.
- *sentiment*: indicator of whether it’s positive (1) or negative (0).
A few samples of tweets text from the dataset and relative sentiment are shown in Table 1. As we can see the quality of writing is quite low.
![Table 1](https://github.com/MatteoM95/Twitter-Sentiment-Analisys/blob/master/Images/TableI.png "Table 1")
![Fig. 1](https://github.com/MatteoM95/Twitter-Sentiment-Analisys/blob/master/Images/WordCloudNegative.svg "Fig. 2")
![Fig. 2](https://github.com/MatteoM95/Twitter-Sentiment-Analisys/blob/master/Images/WordCloudPositive.svg "Fig. 2")

The data-set is divided into two parts:
- a *development set*, containing 224994 entries for which, in addition to the previously mentioned features, the sentiment of the tweet is also known.
- an *evaluation set*, comprised of 74999 entries. We will use the development set to build a classification model to predict the sentiment of these tweets.
Let us now focus our attention on the development set.
In Figure 3, the distribution of the sentiment feature is shown and we can observe that there are more positive tweets
overall implying that the dataset is unbalanced.
One more thing that is worth mentioning is the distribution of the sentiment over the date attribute. All the tweets in the dataset are posted in a period between April and June 2009. Initially, we assumed that the date did not provide valuable information in predicting the sentiment of tweets, but as shown in figure 4, it plays an important role. For instance, people tend to leave more positive comments during specific hours in a day. Also, different days of week and month give us different portions of negative and positive tweets which can be explained by the general mood of society or important events.
Moreover, the wordclouds shown in Figures 1 and 2 give us a visual representation of the word distribution of positive and negative sentiment tweets.

![Fig. 3](https://github.com/MatteoM95/Twitter-Sentiment-Analisys/blob/master/Images/DistributionOfSentiment.svg "Fig. 3")
![Fig. 4](https://github.com/MatteoM95/Twitter-Sentiment-Analisys/blob/master/Images/dateDistributionHour.svg "Fig. 4")
