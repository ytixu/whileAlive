from collections import Counter
import numpy as np
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn import svm

# Check Wordnet version
print 'Using WordNet version', wn.get_version()

# import nltk import download
# download('stopwords')

STOP_WORDS = set(stopwords.words('english'))

def remove_stopwords(context):
    return [c for c in context if c not in STOP_WORDS]

def wordnet_sense(word, pos=None):
	synsets = wn.synsets(word)
	if pos:
		synsets = [ss for ss in synsets if str(ss.pos()) == pos]
	return synsets

def is_good_prediction(predicted_ss, true_ss):
	return len(set(predicted_ss) & set(true_ss)) > 0

def get_overlap(context, ss):
	return len(context.intersection(ss.definition().lower().split()))

def __best_wordnet_sense(wsd_instance, with_pos=False, get_score=False):
	pos = None
	if with_pos:
		pos = wsd_instance.pos

	synsets = wordnet_sense(wsd_instance.lemma, pos)

	if len(synsets) == 0:
		if get_score:
			return [], 0, None
		return []

	keys = [l._key for l in synsets[0].lemmas()]

	if get_score:
		return keys, get_overlap(set(wsd_instance.context), synsets[0]), synsets[0]
	return keys

def __best_wordnet_sense_with_pos(wsd_instance):
	return __best_wordnet_sense(wsd_instance, True)

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
	return [l._key for l in synset.lemmas()]

def __nltk_lest_sense_no_stropwords(wsd_instance):
	return __nltk_lesk_sense(wsd_instance, True)

def __nltk_lest_sense_with_pos(wsd_instance):
	return __nltk_lesk_sense(wsd_instance, False, True)

def __nltk_lest_sense_with_pos_no_stropwords(wsd_instance):
	return __nltk_lesk_sense(wsd_instance, True, True)


# modified lesk

def __combined_lesk_sense(wsd_instance, no_stopword=False, with_pos=False, get_score=False):
	wsd_content = wsd_instance.context
	if no_stopword:
		wsd_content = remove_stopwords(wsd_content)

	context = set(wsd_instance.context)

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

	synset = None
	if with_pos:
		synset = lesk(context, wsd_instance.lemma, pos=wsd_instance.pos)
	else:
		synset = lesk(context, wsd_instance.lemma)

	if synset is None:
		if get_score:
			return [], 0, 0, 0
		return []

	keys = [l._key for l in synset.lemmas()]
	if get_score:
		return keys, get_overlap(set(wsd_instance.context), synset), get_overlap(context, synset), context
	return keys

def __combined_lesk_sense_no_stopwords(wsd_instance):
	return __combined_lesk_sense(wsd_instance, True)

def __combined_lesk_sense_with_pos(wsd_instance):
	return __combined_lesk_sense(wsd_instance, False, True)

def __combined_lesk_sense_no_stopwords_with_pos(wsd_instance):
	return __combined_lesk_sense(wsd_instance, True, True)

# best baseline and model
def best_wordnet_sense(wsd_instance):
	return __best_wordnet_sense(wsd_instance, with_pos=True, get_score=True)

def combined_lesk_sense(wsd_instance):
	return __combined_lesk_sense(wsd_instance, no_stopword=False, with_pos=True, get_score=True)

def vector(wsd_instance):
	wn_sskey, wn_s, wn_ss = best_wordnet_sense(wsd_instance)
	lesk_sskey, lesk_s, lesk_cs, lesk_context = combined_lesk_sense(wsd_instance)
	wn_cs = get_overlap(lesk_context, wn_ss)

	return [lesk_s, wn_s, lesk_cs, wn_cs,\
		len(wsd_instance.context), len(lesk_context)], lesk_sskey, wn_sskey


def train(Ydata, Xdata):
	X = []
	Y = []

	wrong = 0.0
	total = 0.0

	for i, wsd_instance in Xdata.iteritems():
		x, lesk_sskey, wn_sskey = vector(wsd_instance)

		X.append(x)
		if not is_good_prediction(lesk_sskey, Ydata[i]):
			if is_good_prediction(wn_sskey, Ydata[i]):
				Y.append(1)
			else:
				Y.append(1)
				wrong += 1.0
		else:
			Y.append(0)

		total += 1.0
	print 'Bad predictions', wrong/total

	best_model = None
	best_score = 0
	for model in [LogisticRegression, svm.SVC]:
		lg = model()
		lg.fit(X,Y)
		s = lg.score(X,Y)
		if s > best_score:
			best_score = s
			best_model = lg
		print 'Model score', lg.score(X,Y)

	return lg



def predict(wsd_instance, lg):
	x, lesk_sskey, wn_sskey = vector(wsd_instance)

	if lg.predict([x]) == 1:
		lesk_sskey = wn_sskey

	return lesk_sskey

