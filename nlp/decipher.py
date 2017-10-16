import sys, os, glob
import string
import re
import numpy as np
import nltk.tag.hmm as HMM
from nltk.probability import LaplaceProbDist

TEST = 'test_'
TRAIN = 'train_'
CIPHER = 'cipher'
GROUNDTRUTH = 'plain'
ext = '.txt'

states = list(string.ascii_lowercase + ' ,.')

def clean(st):
	return re.sub('[\n\r]', '', st)

def lm_clean(st):
	st = re.sub('[^a-z_ ,\.]', '', st.strip())
	st = re.sub('  ', ' ', st)
	st = re.sub(' \.', '.', st)
	return re.sub(' ,', ',', st)

def get_ass1_counts():
	data_files = glob.glob('rt-polaritydata/*')
	counts = {}
	for filename in data_files:
		data = open(filename, 'r').readlines()
		data = [[x for x in lm_clean(line)] for line in data]
		for line in data:
			for i, char in enumerate(line):
				if i+1 == len(line):
					break
				if char not in counts:
					counts[char] = [{line[i+1]: 1}, 1]
				elif line[i+1] not in counts[char][0]:
					counts[char][0][line[i+1]] = 1
					counts[char][1] += 1
				else:
					counts[char][0][line[i+1]] += 1
					counts[char][1] += 1
	return counts


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

		train_data = [[(clean(x), clean(Y_train[i][j])) for j,x in enumerate(X_line)] for i, X_line in enumerate(X_train)]
		test_data = [[(clean(x), clean(Y_test[i][j])) for j,x in enumerate(X_line)] for i, X_line in enumerate(X_test)]
		return (train_data, X_test, test_data), flag

	except:
		print 'decipher.py [-laplace] [-lm] <cipher_folder>'
		sys.exit(0)


def standard_hmm(data, smoothing=False, lm=None):
	train_data, X_test, test_data = data
	estimator = None
	if smoothing:
		estimator = lambda fdist, bins: LaplaceProbDist(fdist, bins)

	model = HMM.HiddenMarkovModelTrainer(states, states).train_supervised(train_data, estimator=estimator)

	if lm is not None:
		for given_char, params in lm.iteritems():
			freqs, N = params[0], params[1]
			for char, freq in freqs.iteritems():
				n = freq + model._transitions[given_char]._freqdist[char]
				model._transitions[given_char]._freqdist.__setitem__(char, n)

	model.test(test_data)

	for x in X_test:
		predict = model.tag(x)
		print ''.join([i for _,i in predict])

	# for key in model._transitions.keys():
	# 	for i in model._transitions[key]._freqdist.keys():
	# 		# print key, i, model._transitions[key]._freqdist[i], model._transitions[key]._freqdist.N()
	# 		print key, i, model._transitions[key].logprob(i)


if __name__ == '__main__':
	data, flag = main()
	if flag:
		if len(flag) > 1:
			standard_hmm(data, smoothing=True, lm=get_ass1_counts())
		elif flag[0] == '-lm':
			standard_hmm(data, lm=get_ass1_counts())
		else:
			standard_hmm(data, smoothing=True)

	else:
		standard_hmm(data)
