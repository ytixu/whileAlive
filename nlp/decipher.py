import sys, os
import string
import numpy as np
import nltk.tag.hmm as HMM
from nltk.probability import LaplaceProbDist

TEST = 'test_'
TRAIN = 'train_'
CIPHER = 'cipher'
GROUNDTRUTH = 'plain'
ext = '.txt'

states = list(string.ascii_lowercase + ' ,.')

def main():
	args = sys.argv
	try:
		cipher_folder = args[-1]
		flag = None
		if (len(args) >= 3):
			flag = args[1:-1]

		data_files = os.path.join(cipher_folder, TRAIN+CIPHER+ext)
		X_train = open(data_files, 'r').readlines()
		data_files = os.path.join(cipher_folder, TRAIN+GROUNDTRUTH+ext)
		Y_train = open(data_files, 'r').readlines()
		data_files = os.path.join(cipher_folder, TEST+CIPHER+ext)
		X_test = open(data_files, 'r').readlines()
		data_files = os.path.join(cipher_folder, TEST+GROUNDTRUTH+ext)
		Y_test = open(data_files, 'r').readlines()

		train_data = [[(x, Y_train[i][j]) for j,x in enumerate(X_line)] for i, X_line in enumerate(X_train)]
		test_data = [[(x, Y_test[i][j]) for j,x in enumerate(X_line)] for i, X_line in enumerate(X_test)]
		return (train_data, X_test, test_data), flag

	except:
		print 'decipher.py [-laplace] [-lm] <cipher_folder>'

def standard_hmm(data, smoothing=False, lm=False):
	train_data, X_test, test_data = data
	estimator = None
	if smoothing:
		estimator = lambda fdist, bins: LaplaceProbDist(fdist, bins)

	model = HMM.HiddenMarkovModelTrainer(states, states).train_supervised(train_data, estimator=estimator)
	model.test(test_data)

	for x in X_test:
		predict = model.tag(x)
		print ''.join([i for _,i in predict])


if __name__ == '__main__':
	data, flag = main()
	if flag:
		pass
		if len(flag) > 1:
			standard_hmm(data, smoothing=True, lm=True)
		elif flag[0] == '-lm':
			standard_hmm(data, lm=True)
		else:
			standard_hmm(data, smoothing=True)

	else:
		standard_hmm(data)