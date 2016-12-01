import argparse
import sys
import random


from Bio import SeqIO

from PairewiseAlign import Hirschberge, readfq, getHandle, match, compute_match_score, compute_sub_score
import warnings
warnings.filterwarnings("ignore")


D_TH = 1.0
W_MIN = 8
W_MAX = 8
NUCS = ['A','G','T','C']
HIT_SIZE = 1

def _progress_bar(update):
	sys.stdout.write('\r')
	sys.stdout.write("[%d]" % (update))
	sys.stdout.flush()

def get_combo(w, p):
	if len(w) == 1:
		if p[0] < D_TH:
			return NUCS
		return [w]

	combos = get_combo(w[1:],p[1:])
	if p[0] < D_TH:
		return [n+c for n in NUCS for c in combos]
	return [w[0]+c for c in combos]

def match_score(query, ref, prob):
	score = 0
	for i, a in enumerate(query):
		if a == ref[i]:
			score += compute_match_score(ref[i], a, prob[i])
		else:
			score += compute_sub_score(ref[i], a, prob[i])

	return score


class Blast:

	def __init__(self, refseq, prob):

		import pickle
		import itertools
		import numpy as np

		# build dictionary
		self.refseq = refseq
		self.size = len(refseq)
		self.prob = prob
		n = len(refseq)
		print n
		for w_len in range(W_MIN,W_MAX+1):
			# self.database = pickle.load(open('database-'+str(w_len)+'.p', 'r'))
			self.database = {}
			for i in range(n-w_len+1):
				_progress_bar(i*100.0/(n-w_len-1))
				w = refseq[i:w_len+i]
				for c in map(str, get_combo(w, prob[i:w_len+i])):
					score = int(match_score(c, w, prob[i:w_len+i]))
					# if score > MIN_HIT_SCORE:
					if c not in self.database:
						self.database[c] = [(i, score)]
					else:
						self.database[c].append((i, score))
			pickle.dump(self.database, open('database-'+str(w_len)+'.p', 'wb'))

			print ''
			print w_len
			print len(self.database.keys())
			hit_size = map(len, self.database.values())
			print np.mean(hit_size)
			print np.std(hit_size)

			counts = {} 
			for w in self.database.values():
				for _, s in w:
					if s not in counts:
						counts[s] = 1
					else:
						counts[s] += 1
			print counts

	def _dump_hirshberge(self, ref, qry, prob, rev = False):
		row, column, middle, score = Hirschberge(ref, qry, prob)
		if rev:
			self.row_rev = row[::-1]
			self.middle_rev = middle[::-1]
			self.column_rev = column[::-1]
		else:
			self.row = row
			self.middle = middle
			self.column = column
		return score

	def dump_alignment(self):
		print '---------------'
		print self.hits

		if 'row' in self.__dict__:
			print self.row
			print self.middle
			print self.column

		print self.hit_word
		print self.hit_match

		if 'row_rev' in self.__dict__:
			print self.row_rev
			print self.middle_rev
			print self.column_rev

	def search(self, query, w_size, min_hit_score):
		# print self.database.keys()
		max_score = 0
		best_align = -1000
		best_align = -1000
		l = len(query)

		for i in range(0,l-w_size+1):
			w = query[i:w_size+i]
			if w in self.database:
				for hit, s in self.database[w]:
					if s < min_hit_score:
						continue
					# print '----------------'
					score = 0
					low = max(0,hit-i*HIT_SIZE)
					ref = self.refseq[low:hit][::-1]
					qry = query[:i][::-1]
					if len(qry) > 0:
						prob = self.prob[low:hit][::-1]
						score = self._dump_hirshberge(ref, qry, prob, True)

					ref_match = self.refseq[hit:w_size+hit]
					score += match_score(w, ref_match, self.prob[hit:w_size+hit])
					self.hit_word = w
					self.hit_match = ref_match

					high = min(self.size,hit+w_size+(l-i-w_size)*HIT_SIZE)
					ref = self.refseq[hit+w_size:high]
					qry = query[i+w_size:]
					if len(qry) > 0:
						prob = self.prob[hit+w_size:high]
						score += self._dump_hirshberge(ref, qry, prob)

					if score > max_score:
						self.hits = [hit for hit, s in self.database[w] if s > min_hit_score]
						max_score = score
						best_align = hit-i

		if best_align < 0:
			return None
		return max_score, best_align


def alter(w, s, diff=2, gap=1):
	for i in range(diff):
		index = random.randint(0, s)
		w = w[:index]+random.choice('AGTC')+w[index+1:]
	for i in range(gap):
		index = random.randint(0, s-1)
		w = w[:index]+w[index+1:]
	return w

def rand_query(ref, s, n, diff=2, gap=1):
	l = len(ref)
	w = []
	for i in range(n):
		start = random.randint(0,l-s-1)
		new_w = ref[start:start+s]
		w.append((start, alter(new_w, s, diff, gap)))
	return w

def build_solution(ref, prob, s):
	l = len(ref)
	soln = {}
	for i in range(l-s+1):
		w = ref[i:i+s+1]
		p = prob[i:i+s+1]
		combos = get_combo(w, p)
		for c in combos:
			score = match_score(c, w, p)
			if c not in soln or soln[c][0] < score:
				soln[c] = (score, i)

	return soln

def rand_in_dict(soln):
	for key in soln.keys():
		yield key


if __name__ == '__main__':
	o = sys.stdout
	e = sys.stderr
	parser= argparse.ArgumentParser(
		description="This program can return the alignment for two sequences in by only using linear space. " +
		"The sequence files can be either in FASTA or FASTQ format")
	parser.add_argument("file1", help="reference sequence file <Must be in FASTA/Q> format")
	parser.add_argument("file2", help="reference confidence file <Must be in TXT/Q> format")
	args = parser.parse_args() 
	reference = fasta_sequences = [f.seq for f in SeqIO.parse(getHandle(args.file1),'fasta')][0]
	# seqstr2 = list(readfq(getHandle(args.file2)))[0][1]
	probabilities = map(float, getHandle(args.file2).readline().strip().split(' '))
	
	size = 10
	sample = 1000
	diff = 2
	gap = 1

	l = len(reference)
	# soln = build_solution(reference, probabilities, size)

	blast = Blast(reference, probabilities)

	print 'Building queries'
	queries = rand_query(reference, size, sample, diff, gap)

	for MIN_HIT_SCORE in range(-13, 14):
		suboptimal = 0
		no_hit = 0
		wrong_hit = 0
		print '>>> hit score > ', MIN_HIT_SCORE

		for count, query_pair in enumerate(queries):
			i, q = query_pair
			_progress_bar(count)
			# print q, i

		# for q in rand_in_dict(soln):
			predict = blast.search(q, W_MIN, MIN_HIT_SCORE)
			if not predict:
				# print 'No hit.'
				no_hit += 1
			else:
				# print predict
				# true_score = match_score(q, reference[predict[1]:max(predict[1]+len(q), l)], probabilities[predict[1]:max(predict[1]+len(q), l)])
				from_score = match_score(q, reference[i:max(i+len(q), l)], probabilities[i:max(i+len(q), l)])
				if from_score - predict[0] > 0:
					suboptimal += 1
					# print q,i,from_score
					# blast.dump_alignment()
					# print predict
					if sum([1 if abs(i-a) > gap else 0 for a in blast.hits]) > 0:
						wrong_hit += 1
					# print true_score
			# if q in soln:
				# print q
				# if abs(soln[q][1] - predict[1]) > 0:
				# 	blast.dump_alignment()
				# 	print predict
				# 	print soln[q]
				# 	print match_score(q, reference[predict[1]:size+1], probabilities[predict[1]:size+1])
		print ''
		print 'Suboptimal ', suboptimal
		print 'Wrong hit ', wrong_hit
		print 'No hit ', no_hit
