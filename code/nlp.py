# Common imports
import numpy as np
import os
import pandas as pd

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
#%matplotlib notebook
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from gensim.utils import simple_preprocess

import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings = lambda *a, **kw: None

# to make this notebook's output identical at every run
np.random.seed(42)

# gensim: This package contains interfaces and functionality to compute pair-wise document similarities within a corpus of documents.
# gensim.utils.simple_preprocess: Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long.
# unidecode.unidecode: function for removing unicode

# data imports and transformation
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups() # object is a dictionary
W, y = data.data, data.target
df = pd.DataFrame(W, columns=['text'])
df['topic'] = y


# Count words per document.
def get_words_per_doc(txt):
    # split text into words and count them.
    return len(txt.split())

# Build a frequency distribution over words with `Counter`.
from collections import Counter
freqs = Counter()
for i, row in df.iterrows():
    freqs.update(row['text'].lower().split())
    if i > 100:
        break
freqs.most_common()[:20] # can use most frequent words as style/function words

# Sentiment analysis using nltk
document = W[0]
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
polarity = sid.polarity_scores(document) # where document is a string

# draw random sample
dfs = df.sample(frac=0.2, random_state = 42)

# apply compound sentiment score to data-frame
def get_sentiment(snippet):
    return sid.polarity_scores(snippet)['compound']
dfs['sentiment'] = dfs['text'].apply(get_sentiment)

# what do the sentiment scores ‘neg’, ‘neu’,’ pos’ and ‘compound’ stand for?


# StopWords
# There is also a corpus of stopwords, that is, high-frequency words like the, to and also that we sometimes want to filter out of a  document
# before further processing. Stopwords usually have little lexical content, and their presence in a text fails to distinguish it from other texts.

from nltk.corpus import stopwords
# set() converts iterable to set class
stopwords = set(stopwords.words('english'))

# regex
import re

docs = dfs[:5]['text']

# Extract words after Subject.
for doc in docs:
    print(re.findall(r'Subject: \w+ ', # pattern to match. always put 'r' in front of string so that backslashes are treated literally.
                     doc,            # string
                     re.IGNORECASE))  # ignore upper/lowercase (optional)

# Extract hyphenated words
for doc in docs:
    print(re.findall(r'[a-z]+-[a-z]+',
                     doc,
                     re.IGNORECASE))

# extract email addresses
for i, doc in enumerate(docs):
    finder = re.finditer('\w+@.+\.\w\w\w', # pattern to match ([^\s] means non-white-space)
                     doc)            # string
    for m in finder:
        print(i, m.span(),m.group()) # location (start,end) and matching string

# match patterns
pattern1 = r'(\b)uncertain[a-z]*'
pattern2 = r'(\b)econom[a-z]*'
pattern3 = r'(\b)congress(\b)|(\b)deficit(\b)|(\b)federal reserve(\b)|(\b)legislation(\b)|(\b)regulation(\b)|(\b)white house(\b)'

re.search(pattern1,'The White House tried to calm uncertainty in the markets.')

def indicates_uncertainty(doc):
    m1 = re.search(pattern1, doc, re.IGNORECASE)
    m2 = re.search(pattern2, doc, re.IGNORECASE)
    m3 = re.search(pattern3, doc, re.IGNORECASE)
    if m1 and m2 and m3:
        return True
    else:
        return False

# Featurizing text
text = "Prof. Zurich hailed from Zurich. She got 3 M.A.'s from ETH."

# tokenise
from nltk import sent_tokenize
sentences = sent_tokenize(text)

# spacy tends to make less errors than nltk, but is slower
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
sentences = list(doc.sents)
print(sentences)

# Capitalization
text_lower = text.lower() # go to lower-case
text_lower

# recipe for fast punctuation removal
from string import punctuation
punc_remover = str.maketrans('','',punctuation)
text_nopunc = text_lower.translate(punc_remover)
print(text_nopunc)

# splits a string on white space
tokens = text_nopunc.split()

# Numbers
# remove numbers (keep if not a digit)
no_numbers = [t for t in tokens if not t.isdigit()]
# keep if not a digit, else replace with "#"
norm_numbers = [t if not t.isdigit() else '#'
                for t in tokens ]
print(no_numbers )
print(norm_numbers)

# keep if not a stopword
from nltk.corpus import stopwords
stoplist = stopwords.words('english')
nostop = [t for t in norm_numbers if t not in stoplist]
print(nostop)

# scikit-learn stopwords
# from sklearn.feature_extraction import stop_word

# Word Inheritance: stemming
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('english') # snowball stemmer, english

# remake list of tokens, replace with stemmed versions
tokens_stemmed = [stemmer.stem(t) for t in ['tax','taxes','taxed','taxation']]

# Lemmatizing
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
[wnl.lemmatize(c) for c in ['corporation', 'corporations', 'corporate']]

# Pre-processing strategy
from string import punctuation
translator = str.maketrans('','',punctuation)
from nltk.corpus import stopwords
stoplist = set(stopwords.words('english'))
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')

def normalize_text(doc):
    "Input doc and return clean list of tokens"
    doc = doc.replace('\r', ' ').replace('\n', ' ')
    lower = doc.lower() # all lower case
    nopunc = lower.translate(translator) # remove punctuation
    words = nopunc.split() # split into tokens
    nostop = [w for w in words if w not in stoplist] # remove stopwords
    no_numbers = [w if not w.isdigit() else '#' for w in nostop] # normalize numbers
    stemmed = [stemmer.stem(w) for w in no_numbers] # stem each word
    return stemmed

# Word tagging
nltk.download('averaged_perceptron_tagger')
from nltk.tag import perceptron
from nltk import word_tokenize
tagger = perceptron.PerceptronTagger()
tokens = word_tokenize(text)
tagged_sentence = tagger.tag(tokens)
tagged_sentence

# plot nouns and adjectives by topic
from collections import Counter
from nltk import word_tokenize

def get_nouns_adj(snippet):
    tags = [x[1] for x in tagger.tag(word_tokenize(snippet))]
    num_nouns = len([t for t in tags if t[0] == 'N']) / len(tags)
    num_adj = len([t for t in tags if t[0] == 'J']) / len(tags)
    return num_nouns, num_adj

dfs['nouns'], dfs['adj'] = zip(*dfs['text'].map(get_nouns_adj))
dfs.groupby('topic')[['nouns','adj']].mean().plot()

# Spacy corpus prep (https://spacy.io/)
# pip install spacy
# python -m spacy download en_core_web_smimport spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Process whole documents
text = ("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")

doc = nlp(text)

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)

# tokens
list(doc)

# lemmas
[x.lemma_ for x in doc]

# POS tags
[x.tag_ for x in doc]

# N-grams
from nltk import ngrams
from collections import Counter

# get n-gram counts for 10 documents
grams = []
for i, row in df.iterrows():
    tokens = row['text'].lower().split() # get tokens
    for n in range(2,4):
        grams += list(ngrams(tokens,n)) # get bigrams, trigrams, and quadgrams
    if i > 50:
        break
Counter(grams).most_common()[:8]  # most frequent n-grams

# Tokenizers
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(min_df=0.001, # at min 0.1% of docs
                        max_df=.8,
                        max_features=1000,
                        stop_words='english',
                        ngram_range=(1,3))
X = vec.fit_transform(df['text'])

# save the vectors
# pd.to_pickle(X,'X.pkl')

# save the vectorizer
# (so you can transform other documents,
# also for the vocab)
#pd.to_pickle(vec, 'vec-3grams-1.pkl')

# tf-idf vectorizer up-weights rare/distinctive words
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=0.001,
                        max_df=0.9,
                        max_features=1000,
                        stop_words='english',
                        use_idf=True, # the new piece
                        ngram_range=(1,2))

X_tfidf = tfidf.fit_transform(df['text'])
#pd.to_pickle(X_tfidf,'X_tfidf.pkl')
vocab = tfidf.get_feature_names()

# hash vectorizer
from sklearn.feature_extraction.text import HashingVectorizer

hv = HashingVectorizer(n_features=10)
X_hash = hv.fit_transform(df['text'])
X_hash

# Feature selection
# Univariate feature selection using chi2
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression, f_classif, mutual_info_classif
select = SelectKBest(chi2, k=10)
Y = df['topic']==1
X_new = select.fit_transform(X, Y)
# top 10 features by chi-squared:
[vocab[i] for i in np.argsort(select.scores_)[-10:]]

# Document distance
# compute pair-wise similarities between all documents in corpus"
from sklearn.metrics.pairwise import cosine_similarity

sim = cosine_similarity(X[:100])
sim.shape

# TF-IDF Similarity
tsim = cosine_similarity(X_tfidf[:100])
tsim[:4,:4]