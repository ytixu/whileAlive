import numpy as np
import nltk
from nltk.corpus import stopwords

FILES = ['rt-polaritydata/rt-polarity.neg', 'rt-polaritydata/rt-polarity.pos']

def get_data(filename):
	with open(filename, 'rU') as f:
		for line in f:
			tokens = nltk.word_tokenize(line)
			unigrams = [token for token in tokens if token not in stopwords.words('english')]
			bigrams = ngrams(unigrams, 2)
			yield (unigrams, bigrams)

class SentimentClassifier:

	def __init__(self):
		self.p = {}
		self.lexicon = {}

