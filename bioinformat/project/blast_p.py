import argparse
import sys
import random
from Bio import SeqIO

from PairewiseAlign import Hirschberge, readfq, getHandle, match, compute_match_score, compute_sub_score

D_TH = 0.9
W_MIN = 7
W_MAX = 13
NUCS = ['A','G','T','C']
HIT_SIZE = 2

def get_combo(w, p):
	if len(w) == 1:
		if p[0] < D_TH:
			return NUCS
		return [w]

	combos = get_combo(w[1:],p[1:])
	if p[0] < D_TH:
		return [n+c for n in NUCS for c in combos]
	return [w[0]+c for c in combos]


class Blast:

	def __init__(self, refseq, prob):
		# build dictionary
		self.database = {}
		self.refseq = refseq
		self.size = len(refseq)
		self.prob = prob
		n = len(refseq)
		for w_len in range(W_MIN,W_MAX+1):
			for i in range(n-w_len):
				w = refseq[i:w_len+i]
				for c in map(str, get_combo(w, prob[i:w_len+i])):
					if c not in self.database:
						self.database[c] = [i]
					else:
						self.database[c].append(i)

	def _dump_hirshberge(self, ref, qry, prob, rev = False):
		row, column, middle, score = Hirschberge(ref, qry, prob)
		if rev:
			print row[::-1]
			print middle[::-1]
			print column[::-1]
		else:
			print row
			print middle
			print column
		return score

	def match_score(self, query, ref, prob):
		score = 0
		for i, a in enumerate(query):
			if a == ref[i]:
				score += compute_match_score(ref[i], a, prob[i])
			else:
				score += compute_sub_score(ref[i], a, prob[i])

		return score


	def search(self, query, w_size):
		# print self.database.keys()
		max_score = 0
		best_align = 0
		l = len(query)

		for i in range(0,l-w_size+1):
			w = query[i:w_size+i]
			print w, i
			if w in self.database:
				for hit in self.database[w]:
					print '----------------'
					score = 0
					low = max(0,hit-i*HIT_SIZE)
					print low, i, hit, HIT_SIZE, w_size
					ref = self.refseq[low:hit][::-1]
					qry = query[:i][::-1]
					if len(qry) > 0:
						prob = self.prob[low:hit][::-1]
						score = self._dump_hirshberge(ref, qry, prob, True)

					ref_match = self.refseq[hit:w_size+hit]
					score += self.match_score(w, ref_match, self.prob[hit:w_size+hit])
					print w, 'hit'
					print ref_match

					high = min(self.size,hit+w_size+(l-i-w_size)*HIT_SIZE)
					ref = self.refseq[hit+w_size:high]
					qry = query[i+w_size:]
					if len(qry) > 0:
						prob = self.prob[hit+w_size:high]
						score += self._dump_hirshberge(ref, qry, prob)

					print score

					if score > max_score:
						max_score = score
						best_align = hit

		return max_score, best_align


def alter(w, s, diff=2, gap=1):
	for i in range(diff):
		index = random.randint(0, s)
		w = w[:index]+random.choice('AGTC')+w[index+1:]
	for i in range(gap):
		index = random.randint(0, s)
		w = w[:index]+w[index+1:]
	return w

def rand_query(ref, s=40, n=10):
	w = {}
	l = len(ref)
	for i in range(n):
		start = random.randint(0,l-s-1)
		new_w = ref[start:start+s+1]
		w[start] = alter(new_w, s, 0, 0)

	return w


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
	size = 8
	blast = Blast(reference, probabilities)
	queries = rand_query(reference, size, 1)
	for i, q in queries.iteritems():
		print q, i
		print blast.search(q, 7)



# for i in range(100):
#   x = random.randint(0, n-40)
#   y = random.randint(30, 40)
#   query = seqstr1[max(0,x-2):x+y-15] + seqstr1[x+y-5:x+y-2]
#   row, column, middle = Hirschberge(seqstr1[x:x+y], query, seqstr2[x:x+y])
#   print '#' * 8, "Alignment %r" % i, '#' * 8
#   print row
#   print middle
#   print column
