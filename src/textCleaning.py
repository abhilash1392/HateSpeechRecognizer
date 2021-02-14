# src/textCleaning.py 
from nltk import tokenize
import numpy as np
import pandas as pd 
import string 
import re
import nltk 
from nltk.tokenize import TweetTokenizer



def clean_text(text):
    # Removing the punctutation
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation ])
    # Removing the numbers
    text_rc = re.sub('[0-9]+','',text_lc)
    # tokenizing using TweetTokenizer
    tokens = TweetTokenizer(strip_handles=True).tokenize(text_rc)
    # Lemmatizing and removing stopwords
    wn = nltk.WordNetLemmatizer()
    stopword = nltk.corpus.stopwords.words('english')
    text = [wn.lemmatize(word) for word in tokens if word not in stopword]
    return text

