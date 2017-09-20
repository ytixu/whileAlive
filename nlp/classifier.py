import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.classify.naivebayes import NaiveBayesClassifier
from sklearn.model_selection import KFold

FILES = 'rt-polaritydata/rt-polarity.'

def get_data(filename):
	return open(filename, 'rU').read().split('\n'):

class SentimentClassifier:

	def __init__(self):
		self.classifier = None
		self.test_pos = None
		self.test_neg = None

		self.stop_words = set(stopwords.words('english'))
		self.stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])


	def get_next_fold(self, X_pos, X_neg, k=5):
		cross_kf = KFold(n_splits=k)
		# get training / testing
		neg_sets = cross_kf.split(X_neg)
		for i, pos_sets in enumerate(cross_kf.split(X_pos)):
			train_pos, test_pos = pos_sets
			train_neg, test_neg = neg_sets[i]

			X_train_pos, X_test_pos = X_pos[train_pos], X_pos[test_pos]
			X_train_neg, X_test_neg = X_neg[train_neg], X_neg[test_neg]

			self.test_pos = X_test_pos
			self.test_neg = X_test_neg

			# get training / validation
			sub_neg_sets = cross_kf.split(X_train_neg)

			for j, sub_pos_sets in cross_kf.split(X_train_pos):
				ttrain_pos, val_pos = sub_pos_sets
				ttrain_neg, val_neg = sub_neg_sets[j]

				X_ttrain_pos, X_val_pos = X_pos[ttrain_pos], X_pos[val_pos]
				X_ttrain_neg, X_val_neg = X_neg[ttrain_neg], X_neg[val_neg]

				x = np.concatenate(X_ttrain_pos, X_ttrain_neg)
				xv = np.concatenate(X_val_pos, X_val_neg)

				y = np.zeros(len(x))
				y[:len(ttrain_pos)] = 'pos'
				y[len(ttrain_pos):] = 'neg'
				yv = np.zeros(len(xv))
				yv[:len(val_pos)] = 'pos'
				yv[len(val_pos):] = 'neg'

				yield x, xv, y, yv

		raise StopIteration

	def preprocess(self, data, label, n_gram=1, filter_stpw=True):
		X = [[]]*len(data)
		for i, entry in enumerate(data):
			if filter_stpw:
				tokens = nltk.word_tokenize(entry)
				unigrams = [token for token in tokens if token not in stopwords.words('english')]
			if n_gram == 1:
				X[i] = (' '.join(unigrams), label[i])
			else:
				X[i] = ngrams(unigrams, 2)

		return X

	def test(data, label):
		for t in data:

	def train(self, y='pos'):
		pos_data = get_data(FILES+'pos')
		neg_data = get_data(FILES+'neg')

		for x, xv, y, yv in self.get_next_fold(pos_data, neg_data):
			# training 
			train = self.preprocess(x, y)
			self.classifier = NaiveBayesClassifier.train(train)
			# validate 

		
			self.test()
