# Yi Tian Xu
# 260520039
# COMP 550 assignment 4

import sys
import re
import copy
import itertools
import numpy as np
from glob import glob
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords')

_lemmatizer = WordNetLemmatizer()
_stopwords = set(stopwords.words('english'))
_stoppuncs = ['\'', '\"', ',', '\.', '\-']

WORD_COUNT = 100

_flatten = lambda x: list(itertools.chain(*x))

def segment(article):
	# sentence segmentation
	lines = sent_tokenize(article)
	# word segmentation and lowercase
	tokens = [word_tokenize(re.sub('['+'|'.join(_stoppuncs)+']', '',line.lower())) for line in lines]
	# lemmatize
	tokens = [[_lemmatizer.lemmatize(w) for w in line] for line in tokens if len(line) > 0]
	return lines, tokens

def remove_stopwords(tokens):
	# remove stopwords
	return [w for w in tokens if w not in _stopwords]

# apply segmentation and remove stopwords
def clean_input(article):
	lines, tokens = segment(article)
	tokens = [remove_stopwords(line) for line in tokens]
	return lines, tokens

# SumBasic algorithm
# Input segmented article (a list of sentences and a list of words for each sentence)
# update_probs option to update the word probability
def sumbasic(sentences, words, update_probs=True):
	w_probs = FreqDist(words[0])
	for tokens in words[1:]:
		w_probs.update(tokens)
	w_probs = {w:np.log(w_probs[w]*1.0/w_probs.N()) for w in w_probs}

	summary = []
	word_count = 0
	while word_count < WORD_COUNT:
		avg_w_prob_by_sent = np.array([np.average([w_probs[w] for w in sent]) for sent in words])
		if update_probs:
			best_sent_idx = np.argsort(-avg_w_prob_by_sent)[0]
			for w in words[best_sent_idx]:
				w_probs[w] = w_probs[w]*2
		else:
			best_sent_idx = np.argsort(-avg_w_prob_by_sent)[len(summary)]

		summary.append(sentences[best_sent_idx])
		word_count += len(sentences[best_sent_idx].split())

	return ' '.join(summary)

# Get the leading sentences of an article
def leading_summary(data_file):
	with open(data_file, 'rU') as article_file:
		leading = []
		sent_idx = 0
		sentences = sent_tokenize(article_file.read())
		while len(leading) < WORD_COUNT:
			leading = leading + sentences[sent_idx].split()
			sent_idx += 1

	return ' '.join(leading)

# ROUGE 1 score, comparing a synthetic summary
# with all Leading summaries in a cluster of article (specified by ref_files)
def rouge_1(ref_files, summary):
	refs = []
	for data_file in ref_files:
		refs.append(remove_stopwords(_flatten(segment(leading_summary(data_file))[1])[:WORD_COUNT]))
	sum_words =  remove_stopwords(_flatten(segment(summary)[1])[:WORD_COUNT])
	score = 0.0

	for ref in refs:
		a = copy.copy(sum_words)
		for ngram in ref:
			try:
				a.remove(ngram)
				score += 1.0
			except:
				pass

	return score/sum([len(ref) for ref in refs])


if __name__ == '__main__':
	args = sys.argv
	try:
		method_name = args[1]
		data_files = []
		for f in args[2:]:
			data_files.extend(glob(f))
	except:
		print 'sumbasic.py <method_name> <file_n>*'
		sys.exit(0)

	summary = None
	if method_name == 'leading':
		# compute average ROUGE 1 score
		avg = 0.0
		n = len(data_files)
		for i in range(n):
			data_files_ = copy.copy(data_files)
			summary = leading_summary(data_files_[i])
			data_files_.remove(data_files_[i])
			avg += rouge_1(data_files_, summary)

		# get random leading summary
		leading_file = np.random.choice(data_files_)
		summary = leading_summary(leading_file)
		print '---average ROUGE-1 score:', avg/n

	else:
		# combine all articles
		sentences = []
		words = []
		for data_file in data_files:
			with open(data_file, 'rU') as article_file:
				lines, tokens = clean_input(article_file.read())
				sentences.extend(lines)
				words.extend(tokens)

		# run SumBasic
		if method_name == 'orig':
			summary = sumbasic(sentences, words)
		elif method_name == 'simplified':
			summary = sumbasic(sentences, words, False)
		else:
			print 'Invalid method name. Either "orig", "simplified" or "leading".'

		print '---ROUGE-1 score:', rouge_1(data_files, summary)

	print summary