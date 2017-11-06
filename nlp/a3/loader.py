
import xml.etree.cElementTree as ET
import codecs
import itertools
from nltk.corpus import stopwords

import word_sense_classifier

# import nltk import download
# download('stopwords')

STOP_WORDS = set(stopwords.words('english'))

class WSDInstance:
    def __init__(self, my_id, lemma, pos, context, index):
        self.id = my_id         # id of the WSD instance
        self.lemma = lemma      # lemma of the word whose sense is to be resolved
        self.pos = pos
        self.context = context  # lemma of all the words in the sentential context
        self.context_no_stopwords = remove_stopwords(context)
        self.index = index      # index of lemma within the context
    def __str__(self):
        '''
        For printing purposes.
        '''
        return '%s\t%s\t%s\t%d' % (self.id, self.lemma, ' '.join(self.context), self.index)

def remove_stopwords(context):
    return [c for c in context if c not in STOP_WORDS]

def clean(word, is_context):
    # lowercase and convert '-' and '_'
    if is_context:
        return word.lower().replace('_', '-')
    return word.lower()

def load_instances(f):
    '''
    Load two lists of cases to perform WSD on. The structure that is returned is a dict, where
    the keys are the ids, and the values are instances of WSDInstance.
    '''
    tree = ET.parse(f)
    root = tree.getroot()

    dev_instances = {}
    test_instances = {}

    for text in root:
        if text.attrib['id'].startswith('d001'):
            instances = dev_instances
        else:
            instances = test_instances
        for sentence in text:
            # construct sentence context
            context = [clean(to_ascii(el.attrib['lemma']), True) for el in sentence]
            for i, el in enumerate(sentence):
                if el.tag == 'instance':
                    my_id = el.attrib['id']
                    lemma = clean(to_ascii(el.attrib['lemma']), False)
                    pos = to_ascii(el.attrib['pos']).lower()[0]
                    instances[my_id] = WSDInstance(my_id, lemma, pos, context, i)
    return dev_instances, test_instances

def load_key(f):
    '''
    Load the solutions as dicts.
    Key is the id
    Value is the list of correct sense keys.
    '''
    dev_key = {}
    test_key = {}
    for line in open(f):
        if len(line) <= 1: continue
        #print line
        doc, my_id, sense_key = line.strip().split(' ', 2)
        if doc == 'd001':
            dev_key[my_id] = sense_key.split()
        else:
            test_key[my_id] = sense_key.split()
    return dev_key, test_key

def to_ascii(s):
    # remove all non-ascii characters
    return codecs.encode(s, 'ascii', 'ignore')

def predict_and_evaluate(keys, dataset, method):
    count = 0.0
    good_predictions = 0.0
    for key, wsd_instance in dataset.iteritems():
        if len(set(method(wsd_instance)) & set(keys[key])) > 0:
            good_predictions += 1.0
        count += 1.0

    return good_predictions/count

if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)

    dev_instances = {k:v for (k,v) in dev_instances.iteritems() if k in dev_key}
    test_instances = {k:v for (k,v) in test_instances.iteritems() if k in test_key}

    methods = [func for name, func in word_sense_classifier.__dict__.iteritems() if callable(func) and name.startswith('__')]
    for method in methods:
        print method.__name__, predict_and_evaluate(dev_key, dev_instances, method)