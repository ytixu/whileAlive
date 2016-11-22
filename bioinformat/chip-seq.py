from Bio import SeqIO ### this is only for reading FASTA file
import math
import sys
import re
import operator

ALPHABET = {'A':0.3, 'C':0.2, 'G':0.2, 'T':0.3, '[A|G]':0.5, '[C|T]':0.5, '\w':1.0}
LENGTH = 5.0

def combinationGenerator():
	for a, pa in ALPHABET.iteritems():
		for b, pb in ALPHABET.iteritems():
			for c, pc in ALPHABET.iteritems():
				for d, pd in ALPHABET.iteritems():
					for e, pe in ALPHABET.iteritems():
						p = pa*pb*pc*pd*pe
						yield (a+b+c+d+e,p)
UNM_TH = 10000000
M_TH = 10000000

class Consensus:

	def __init__(self):
		self.accNwEw = 0
		self.accEw = 0
		self.opp_accNwEw = 0
		self.opp_accEw = 0
		self.unmatched = 0
		self.matched = 0

	def computeScore(self, w, seq, p, good=True):
		e = (len(seq) - LENGTH + 1)*p
		n = len(tuple(re.finditer(r'(?=('+w+'))', seq)))
		if n == 0:
			if good:
				self.unmatched += 1
				if self.unmatched > UNM_TH:
					return True
		elif not good:
			self.matched += 1
			if self.matched > M_TH:
				return True

		if good:
			self.accEw += e
			self.accNwEw += n - e
		else:
			self.opp_accEw += e
			self.opp_accNwEw += n - e

	def dump(self):
		z1 = self.accNwEw/math.sqrt(self.accEw)
		z2 = self.opp_accNwEw/math.sqrt(self.opp_accEw)

		self.accEw = 0
		self.accNwEw = 0
		self.unmatched = 0
		self.opp_accEw = 0
		self.opp_accNwEw = 0
		self.matched = 0
		return (z1,z2)


cons = Consensus()

if len(sys.argv) > 1:
	print 'NOTE: taking %s as the sequences with bounds' % sys.argv[1]
	print 'NOTE: taking %s as the sequences without bounds' % sys.argv[2]
	good_file = open(sys.argv[1])
	bad_file = open(sys.argv[2])
else:
	good_file = open('GATA2_chr1.fa')
	bad_file = open('not_GATA2_chr1.fa')

maxW = -10000
result = {}

for w, p in combinationGenerator():
	notGoodScore = False
	sequences = SeqIO.parse(good_file, 'fasta')
	for fasta in sequences:
		name, sequence = fasta.id, str(fasta.seq)
		if cons.computeScore(w, sequence, p):
			notGoodScore = True

	if not notGoodScore:
		sequences = SeqIO.parse(bad_file, 'fasta')
		for fasta in sequences:
			name, sequence = fasta.id, str(fasta.seq)
			if cons.computeScore(w, sequence, p, False):
				notGoodScore = True

	if not notGoodScore:
		z1, z2 = cons.dump()
		z = z1-z2
		if maxW < z:
			maxW = z
			print 'NEW MAX -----> ', w, z1, z2, z
		else:
			print w, z1, z2, z
		result[w] = z
	
	good_file.seek(0)
	bad_file.seek(0)


print 'top 10~'
print sorted(result.items(), key=operator.itemgetter(1), reverse=True)[0:10]


