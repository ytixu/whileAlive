import numpy as np
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

# Check Wordnet version
print 'Using WordNet version', wn.get_version()

def __best_wordnet_sense(wsd_instance, with_pos=False):
    synsets = wn.synsets(wsd_instance.lemma)
    if with_pos:
        synsets = [ss for ss in synsets if str(ss.pos()) == wsd_instance.pos]

    if len(synsets) == 0:
        return ''
    return [l._key for l in synsets[0].lemmas()]

def __best_wordnet_sense_with_pos(wsd_instance):
    return __best_wordnet_sense(wsd_instance, True)

def __nltk_lesk_sense(wsd_instance, no_stopword=False, with_pos=False):
    context = wsd_instance.context
    if no_stopword:
        context = wsd_instance.context_no_stopwords

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