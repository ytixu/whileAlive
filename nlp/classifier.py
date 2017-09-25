# COMP 550 assigment 1
# Yi Tian Xu
# 260520039

import re
import sys
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.metrics import ConfusionMatrix

nltk.download('stopwords')
nltk.download('punkt')

FILES = 'rt-polaritydata/rt-polarity.'
neg_label = 0
pos_label = 1

def get_data(filename):
	return open(filename, 'rU').read().decode('Latin-1').split('\n')


# This class runs all the classifiers 
# (Naive Bayes, SVM and Logistic Regression)
# and average the confusion matrices and accuracy
# across the 5-fold cross validation
class SentimentClassifier:

	def __init__(self):
		self.stop_words = set(stopwords.words('english'))
		self.stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '/', '\\'])

		self.avg_confusion_mat = {}
		self.accuracy = {}

	# Helper function for printing command line counter
	def print_counter(self, count=0, total=0):
		if total == 0:
			print 'Done'
		else:
			sys.stdout.write("\rProcessing data [%d/%d] " % (count, total))
			sys.stdout.flush()
	
	# Preprocessing for Naive Bayes input
	def n_gram_dict_form(self, data, label, n_gram=1, filter_stpw=False):
		N = len(data)
		X = [()]*N

		for i, entry in enumerate(data):
			self.print_counter(i, N)
			unigrams = nltk.word_tokenize(entry)
			if filter_stpw:
				unigrams = [token for token in unigrams if token not in self.stop_words]
			if n_gram == 1:
				X[i] = ({ngram: True for ngram in unigrams}, label[i])
			else:
				words = {ngram: True for ngram in nltk.ngrams(unigrams, 2)}
				words.update({ngram: True for ngram in unigrams})
				X[i] = (words, label[i])

		self.print_counter()
		return np.array(X)

	# Preprocessing for SVM and Linear Regression input
	def feature_vector_form(self, data, n_gram=1, filter_stpw=False):
		stop_words = None
		ngram_range = (1,1)
		if filter_stpw:
			stop_words = self.stop_words
		if n_gram == 2:
			ngram_range = (1,2)

		vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=ngram_range)
		return vectorizer.fit_transform(data)

	# Average accuracy and confusion matrix
	def update_score(self, exp_type, cm, acc):
		if exp_type not in self.avg_confusion_mat:
			self.avg_confusion_mat[exp_type] = np.zeros((2,2))
			self.accuracy[exp_type] = []

		self.avg_confusion_mat[exp_type][0,0] += cm[str(neg_label),str(neg_label)]
		self.avg_confusion_mat[exp_type][0,1] += cm[str(neg_label),str(pos_label)]
		self.avg_confusion_mat[exp_type][1,0] += cm[str(pos_label),str(neg_label)]
		self.avg_confusion_mat[exp_type][1,1] += cm[str(pos_label),str(pos_label)]
		self.accuracy[exp_type].append(acc)

		print exp_type
		print 'Accuracy %f' % acc
		print(cm.pretty_format(show_percents=True, truncate=9))

	def naive_bayes(self, X, y, yt, train, test, exp_type):
		x, xt = X[train], X[test]

		nb = NaiveBayesClassifier.train(x)
		acc = nltk.classify.accuracy(nb, xt)
		predictions = [nb.classify(t) for t, _ in xt]
		cm = ConfusionMatrix(yt.tolist(), predictions)
		
		self.update_score(exp_type, cm, acc)
		# PRINT MOST INFORMATIVE FEATURES FOR NAIVE BAYES
		# print nb.most_informative_features(35)

	def svm(self, X, y, yt, train, test, exp_type):
		x, xt = X[train], X[test]
		
		clf = svm.SVC(kernel='linear')
		clf.fit(x, y) 
		acc = clf.score(xt, yt)
		predictions = clf.predict(xt)
		cm = ConfusionMatrix(yt.tolist(), predictions.tolist())
		
		self.update_score(exp_type, cm, acc)

	def log_reg(self, X, y, yt, train, test, exp_type):
		x, xt = X[train], X[test]

		lr = LogisticRegression()
		lr.fit(x, y) 
		acc = lr.score(xt, yt)
		predictions = lr.predict(xt)
		cm = ConfusionMatrix(yt.tolist(), predictions.tolist())
		
		self.update_score(exp_type, cm, acc)

	def train(self, k=5):
		# loads corpus
		pos_data = get_data(FILES+'pos')
		neg_data = get_data(FILES+'neg')

		# bring them together with labels into X and Y arrays
		X = np.concatenate(([(x, pos_label) for x in pos_data], [(x, neg_label) for x in neg_data]))
		np.random.shuffle(X)
		Y = np.array([l for _,l in X])
		X = np.array([x for x,_ in X])
		
		# cross validation
		for train, test in list(KFold(len(X), n_folds=k)):
			y, yt = Y[train], Y[test]
			# naive bayes
			X_ = self.n_gram_dict_form(X, Y)
			self.naive_bayes(X_, y, yt, train, test, 'NB-unigram')
			X_ = self.n_gram_dict_form(X, Y, 2, False)
			self.naive_bayes(X_, y, yt, train, test, 'NB-bigram')
			X_ = self.n_gram_dict_form(X, Y, 1, True)
			self.naive_bayes(X_, y, yt, train, test, 'NB-unigram-filtered')
			X_ = self.n_gram_dict_form(X, Y, 2, True)
			self.naive_bayes(X_, y, yt, train, test, 'NB-bigram-filtered')

			# SVM and logistic regression
			X_ = self.feature_vector_form(X)
			self.svm(X_, y, yt, train, test, 'SVM-unigram')
			self.log_reg(X_, y, yt, train, test, 'LOG-unigram')
			X_ = self.feature_vector_form(X, 2, False)
			self.svm(X_, y, yt, train, test, 'SVM-bigram')
			self.log_reg(X_, y, yt, train, test, 'LOG-bigram')
			X_ = self.feature_vector_form(X, 1, True)
			self.svm(X_, y, yt, train, test, 'SVM-unigram-filtered')
			self.log_reg(X_, y, yt, train, test, 'LOG-unigram-filtered')
			X_ = self.feature_vector_form(X, 2, True)
			self.svm(X_, y, yt, train, test, 'SVM-bigram-filtered')
			self.log_reg(X_, y, yt, train, test, 'LOG-bigram-filtered')


		for exp_type, cm in self.avg_confusion_mat.iteritems():
			print exp_type
			print cm
			print np.mean(self.accuracy[exp_type])


# main 
model = SentimentClassifier()
model.train()
