#Importing necessary libraries
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline

#Loading the dataset
data = pd.read_csv('Review.csv')
data.head()

# Downloading NLTK resources
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# User defined functions for text preprocessing

# Function to change case to lowercase
def changecase(text):
    return text.changecase()

# Function to remove non-ASCII characters
def remove_nascii(data_str):
    return ''.join(c for c in data_str if 0 < ord(c) < 127)

# Function to fix common abbreviations
def abbreviations_fix(data_str):
    data_str = data_str.changecase()
    abbreviations = {
        r'\bthats\b': 'that is',
        r'\bive\b': 'i have',
        r'\bim\b': 'i am',
        r'\bya\b': 'yeah',
        r'\bcant\b': 'can not',
        r'\bdont\b': 'do not',
        r'\bwont\b': 'will not',
        r'\bid\b': 'i would',
        r'\bwtf\b': 'what the fuck',
        r'\bwth\b': 'what the hell',
        r'\br\b': 'are',
        r'\bu\b': 'you',
        r'\bk\b': 'OK',
        r'\bsux\b': 'sucks',
        r'\bno+\b': 'no',
        r'\bcoo+\b': 'cool',
        r'\brt\b': '',
    }
    for abbr, replacement in abbreviations.items():
        data_str = re.sub(abbr, replacement, data_str)
    return data_str.strip()

# Function to clean the text (remove URLs, punctuation, numbers)
def clean(data_str):
    url_re = re.compile('https?://(www.)?\w+\.\w+(/\w+)*/?')
    punc_re = re.compile('[%s]' % re.escape(string.punctuation))
    num_re = re.compile('(\\d+)')
    mention_re = re.compile('@(\w+)')
    alpha_num_re = re.compile("^[a-z0-9_.]+$")
    data_str = data_str.changecase()
    data_str = url_re.sub(' ', data_str)
    data_str = mention_re.sub(' ', data_str)
    data_str = punc_re.sub(' ', data_str)
    data_str = num_re.sub(' ', data_str)
    return " ".join(w for w in data_str.split() if alpha_num_re.match(w))

# Function to remove stopwords
def remove_stop_words(data_str):
    stops = set(stopwords.words("english"))
    return " ".join(word for word in data_str.split() if word not in stops)

# Function for part-of-speech tagging
def pos_tag(data_str):
    nn_tags = ['NN', 'NNP', 'NNPS', 'NNS']
    jj_tags = ['JJ', 'JJR', 'JJS']
    vb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    nltk_tags = nn_tags + jj_tags + vb_tags
    tagged_text = nltk.pos_tag(data_str.split())
    return " ".join(word for word, tag in tagged_text if tag in nltk_tags)

# Function for lemmatization
def lemmatize(data_str):
    lmtzr = WordNetLemmatizer()
    tagged_words = nltk.pos_tag(data_str.split())
    lemmatized_text = []
    for word, tag in tagged_words:
        pos = 'v' if tag.startswith('V') else 'n'
        lemmatized_text.append(lmtzr.lemmatize(word, pos))
    return " ".join(lemmatized_text)

# Function to compute sentiment polarity using TextBlob
def sentiment_score(comments):
    return TextBlob(comments).sentiment.polarity

# Dropping rows with null values in required columns
data = data.dropna(subset=['listing_id', 'id', 'date', 'reviewer_id', 'reviewer_name', 'comments'])

# Applying preprocessing functions to 'comments' column
data['comments'] = data['comments'].apply(changecase)
data['comments'] = data['comments'].apply(remove_nascii)
data['comments'] = data['comments'].apply(abbreviations_fix)
data['comments'] = data['comments'].apply(clean)
data['comments'] = data['comments'].apply(remove_stop_words)
data['comments'] = data['comments'].apply(pos_tag)
data['comments'] = data['comments'].apply(lemmatize)

# Applying sentiment analysis to compute sentiment score
data['sentiment_score'] = data['comments'].apply(sentiment_score)
data.head()

# Plotting sentiment score over time
plt.figure(figsize=(15, 8))
plt.scatter(data['date'][:50], data['sentiment_score'][:50])
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Score Over Time')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

# Combine all comments into one text
all_comments = ' '.join(data['comments'])

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_comments)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Comments')
plt.show()

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

# Analyze the sentiment
for comment in data['comments'][:50]:
    result = sentiment_pipeline(comment)
    print(comment)
    print(result)