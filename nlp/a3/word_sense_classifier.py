# COMP 550 - assigment 3
# Nov 13 2017
# Yi Tian Xu
# 260520039

import numpy as np
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.linear_model import LogisticRegression
from sklearn import svm

# Check Wordnet version
print 'Using WordNet version', wn.get_version()

# import nltk import download
# download('stopwords')

STOP_WORDS = set(stopwords.words('english'))

# helper functions

def remove_stopwords(context):
	return [c for c in context if c not in STOP_WORDS]

def is_good_prediction(predicted_ss, true_ss):
	return len(set(predicted_ss) & set(true_ss)) > 0

def get_overlap(context, ss):
	return len(context.intersection(ss.definition().lower().split()))

def get_synset_key(ss):
	return [l._key for l in ss.lemmas()]

def predict_and_evaluate(keys, dataset, method, args=None):
    count = 0.0
    good_predictions = 0.0
    for key, wsd_instance in dataset.iteritems():
        prediction = None
        if args:
            prediction = method(wsd_instance, args)
        else:
            prediction = method(wsd_instance)

        if is_good_prediction(prediction, keys[key]):
            good_predictions += 1.0
        count += 1.0

    return good_predictions/count

# WordNet sense

def wordnet_sense(word, pos=None):
	synsets = wn.synsets(word)
	if pos:
		synsets = [ss for ss in synsets if str(ss.pos()) == pos]
	return synsets

def __best_wordnet_sense(wsd_instance, with_pos=False):
	pos = None
	if with_pos:
		pos = wsd_instance.pos

	synsets = wordnet_sense(wsd_instance.lemma, pos)

	if len(synsets) == 0:
		return []

	keys = get_synset_key(synsets[0])

	return keys

def __best_wordnet_sense_with_pos(wsd_instance):
	return __best_wordnet_sense(wsd_instance, True)

# nltk's lesk

def __nltk_lesk_sense(wsd_instance, no_stopword=False, with_pos=False):
	context = wsd_instance.context
	if no_stopword:
		context = remove_stopwords(context)

	synset = None
	if with_pos:
		synset = lesk(context, wsd_instance.lemma, pos=wsd_instance.pos)
	else:
		synset = lesk(context, wsd_instance.lemma)

	if synset is None:
		return []
	return get_synset_key(synset)

def __nltk_lest_sense_no_stropwords(wsd_instance):
	return __nltk_lesk_sense(wsd_instance, True)

def __nltk_lest_sense_with_pos(wsd_instance):
	return __nltk_lesk_sense(wsd_instance, False, True)

def __nltk_lest_sense_with_pos_no_stropwords(wsd_instance):
	return __nltk_lesk_sense(wsd_instance, True, True)


# modified lesk

def __combined_lesk_sense(wsd_instance, no_stopword=False, with_pos=False, get_list=False):
	wsd_content = wsd_instance.context
	if no_stopword:
		wsd_content = remove_stopwords(wsd_content)

	context = set(wsd_instance.context)

	# augment context
	for i, c_word in enumerate(wsd_instance.context):
		if i == wsd_instance.index:
			continue
		if no_stopword and (c_word not in wsd_content):
			continue

		synsets = None
		if with_pos:
			synsets = wordnet_sense(c_word, wsd_instance.context_pos[i])
		else:
			synsets = wordnet_sense(c_word)

		for ss in synsets:
			context.update(ss.definition().lower().split())

	# get synset using lesk's algo
	synset = None
	if not get_list:
		if with_pos:
			synset = lesk(context, wsd_instance.lemma, pos=wsd_instance.pos)
		else:
			synset = lesk(context, wsd_instance.lemma)

		if synset is None:
			return []

		keys = [l._key for l in synset.lemmas()]
		return keys
	else:
		# this part is for the modified lesk's algo

		# here is same as in nltk's lesk
		synsets = wn.synsets(wsd_instance.lemma)
		if with_pos:
			synsets = [ss for ss in synsets if str(ss.pos()) == wsd_instance.pos]

		# but we return a list of synsets with the augmented context instead
		if not synsets:
			return [], context

		return [(i, len(context.intersection(ss.definition().split())), ss,\
	        	sum([l.count() for l in ss.lemmas()])) for i, ss in enumerate(synsets)], context, synsets


def __combined_lesk_sense_no_stopwords(wsd_instance):
	return __combined_lesk_sense(wsd_instance, True)

def __combined_lesk_sense_with_pos(wsd_instance):
	return __combined_lesk_sense(wsd_instance, False, True)

def __combined_lesk_sense_no_stopwords_with_pos(wsd_instance):
	return __combined_lesk_sense(wsd_instance, True, True)

# best lesk model with POS for modified lesk
def combined_lesk_sense(wsd_instance):
	return __combined_lesk_sense(wsd_instance, no_stopword=False, with_pos=True, get_list=True)

# construct vector for classification
def get_vector(wsd_instance, lesk_ss, lesk_context):
	i, score, ss, freq = lesk_ss
	keys = get_synset_key(ss)
	return keys, [score, freq, i, len(wsd_instance.context), len(lesk_context)]

# this is to count how many samples have negative class prediction
MISS_COUNTS = 0

def train(Ydata, Xdata):
	global MISS_COUNTS

	# construct data matrices
	X = []
	Y = []
	for i, wsd_instance in Xdata.iteritems():
		lesk_sslist, lesk_context, synsets = combined_lesk_sense(wsd_instance)
		true_label = Ydata[i]

		for lesk_ss in lesk_sslist:
			keys, x = get_vector(wsd_instance, lesk_ss, lesk_context)
			X.append(x)

			if is_good_prediction(keys, true_label):
				Y.append(1)
			else:
				Y.append(0)

	# fit model
	best_model = None
	best_score = 0
	for model in [LogisticRegression, svm.SVC]:
		md = model()
		md.fit(X,Y)
		s = md.score(X,Y)
		print md.__class__.__name__, 'fit score', s

		s = predict_and_evaluate(Ydata, Xdata, predict, md)
		print 'Classification score', s

		if s > best_score:
			best_score = s
			best_model = md

		# print MISS_COUNTS
		MISS_COUNTS = 0
	return md


def predict(wsd_instance, lg):
	global MISS_COUNTS
	lesk_sslist, lesk_context, synsets = combined_lesk_sense(wsd_instance)

	X = []
	for lesk_ss in lesk_sslist:
		_, x = get_vector(wsd_instance, lesk_ss, lesk_context)
		X.append(x)

	for i, y in enumerate(lg.predict(X)):
		if y == 1:
			return get_synset_key(lesk_sslist[i][2])

	MISS_COUNTS += 1
	return __best_wordnet_sense_with_pos(wsd_instance)

