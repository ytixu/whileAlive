import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import KFold

FILES = 'rt-polaritydata/rt-polarity.'

def get_data(filename):
	return open(filename, 'rU').read().split('\n'):

class SentimentClassifier:

	def __init__(self):
		self.p = {}
		self.lexicon = {}
		self.test_data = None

	def get_next_fold(self, data, k=5):
		cross_kf = KFold(n_splits=k)
		# get training / testing
		for train, test in cross_kf.split(data):
			X_train, X_test = X[train], X[test]
			self.test_data = X_test
			# get training / validation
			for ttrain, valid in cross_kf.split(X_train):
				X_ttrain, X_valid = X[ttrain], X[valid]
				yield X_ttrain, X_valid

		raise StopIteration

	def preprocess(self, data):
		tokens = nltk.word_tokenize(data)
		unigrams = [token for token in tokens if token not in stopwords.words('english')]
		# bigrams = ngrams(unigrams, 2)
		# yield (unigrams, bigrams)

	def train(self, y='pos'):
		data = get_data(FILES+y)

		for train, valid in self.get_next_fold(data):
			for line in train:
				for word in self.preprocess(line):
					# add probs 

			for line in valid:
				# validate 
				
