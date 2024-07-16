# Sentiment Analysis on Customer-Feedback
This project implements sentiment analysis on customer reviews using various Natural Language Processing (NLP) techniques. The analysis includes text preprocessing, sentiment scoring, and data visualization. Additional NLP methods such as word cloud visualization, sentiment distribution plots, and advanced sentiment analysis using pretrained BERT model is also incorporated.

## Features
- **Text Preprocessing:** Clean and preprocess text data including case normalization, removing non-ascii characters, handling abbreviations, removing punctuations and alphanumeric words, stop word removal, and lemmatization.
- **POS Tagging:** Performed POS tagging on pre processed text.
- **Sentiment Analysis:** Assigning sentiment polarity to each customer review using textblob. Used binary rating as target: 1 for positive and 0 for negative.
- **Word Cloud Visualization:** Generate word clouds to visualize the most frequent words in the reviews.
- **TF-IDF Analysis:** Identify important terms using TF-IDF.
- **Advanced Sentiment Analysis:** Apply pre-trained BERT models for more accurate sentiment analysis.

## Prerequisites

- pandas
- nltk
- textblob
- wordcloud
- transformers

## Usage 
Clone the repository:
```
git clone https://github.com/knowmili/Customer-Feedback.git
cd Customer-Feedback
```

### Install required packages
```
pip install -r requirements.txt
```

### Run the main script
```
python main.py
```




