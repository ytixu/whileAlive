import re
import sys
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import svm
import nltk
from nltk.corpus import stopwords
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.metrics import ConfusionMatrix

# nltk.download('stopwords')
# nltk.download('punkt')

FILES = 'rt-polaritydata/rt-polarity.'

def get_data(filename):
	return open(filename, 'rU').read().split('\n')

class SentimentClassifier:

	def __init__(self):
		self.classifier = None

		self.stop_words = set(stopwords.words('english'))
		# self.stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

		self.avg_confusion_mat = {}
		self.accuracy = {}

	def preprocess(self, data, label, n_gram=1, filter_stpw=False):
		N = len(data)
		X = [()]*N

		for i, entry in enumerate(data):
		
			sys.stdout.write("\rProcessing data [%d/%d] " % (i, N))
			sys.stdout.flush()
		
			unigrams = re.sub(r'\W+ ', ' ', entry)
			unigrams = nltk.word_tokenize(unigrams)
			if filter_stpw:
				unigrams = [token for token in unigrams if token not in self.stop_words]
			if n_gram == 1:
				X[i] = (dict([(ngram, True) for ngram in unigrams]), label[i])
			else:
				X[i] = (dict([(ngram, True) for ngram in nltk.ngrams(unigrams, 2)]), label[i]) 


		print "Done"
		return np.array(X)

	def matrix_form(self, data):
		

	def update_score(self, exp_type, cm, acc):
		if exp_type not in self.avg_confusion_mat:
			self.avg_confusion_mat[exp_type] = np.zeros((2,2))
			self.accuracy[exp_type] = []

		self.avg_confusion_mat[exp_type][0,0] += cm['n','n']
		self.avg_confusion_mat[exp_type][0,1] += cm['n','p']
		self.avg_confusion_mat[exp_type][1,0] += cm['p','n']
		self.avg_confusion_mat[exp_type][1,1] += cm['p','p']
		self.accuracy[exp_type].append(acc)

		print 'Accuracy %f' % acc
		print(cm.pretty_format(show_percents=True, truncate=9))

	def naive_bayes(self, X, Y, train, test, exp_type):
		x, xt = X[train], X[test]
		y, yt = Y[train], Y[test]

		self.classifier = NaiveBayesClassifier.train(x)
		acc = nltk.classify.accuracy(self.classifier, xt)
		predictons = [self.classifier.classify(t) for t, _ in xt]
		cm = ConfusionMatrix(yt.tolist(), predictons)
		
		self.update_score(exp_type, cm, acc)

	def svm(self, X, Y, train, test, exp_type)
		clf = svm.SVC()

		

	def train(self, k=5):
		pos_data = get_data(FILES+'pos')
		neg_data = get_data(FILES+'neg')

		X = np.concatenate(([(x, 'p') for x in pos_data], [(x, 'n') for x in neg_data]))
		np.random.shuffle(X)
		Y = np.array([l for _,l in X])
		X = np.array([x for x,_ in X])
		

		for train, test in list(KFold(len(X), n_folds=k)):
			
			X_ = self.preprocess(X, Y)
			self.naive_bayes(X_, Y, train, test, 'NB-unigram')
			X_ = self.preprocess(X, Y, 2, False)
			self.naive_bayes(X_, Y, train, test, 'NB-bigram')
			X_ = self.preprocess(X, Y, 1, True)
			self.naive_bayes(X_, Y, train, test, 'NB-unigram-filtered')
			X_ = self.preprocess(X, Y, 2, True)
			self.naive_bayes(X_, Y, train, test, 'NB-bigram-filtered')


		for exp_type, cm in self.avg_confusion_mat.iteritems():
			print exp_type
			print cm
			print np.mean(self.accuracy[exp_type])




# main 
model = SentimentClassifier()
model.train()
