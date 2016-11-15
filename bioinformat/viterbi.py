from Bio import SeqIO ### this is only for reading FASTA file
import math
import sys
import csv

HYDROPHOBIC = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
HYDROPHILIC = ['R', 'H', 'K', 'D', 'E', 'T', 'N', 'Q', 'S', 'C', 'G', 'P']

class Stats:

	def __init__(self):
		self.longest_phobic = {'name':'', 'score':0}
		self.big_mix_per = {'name':'', 'score':0.0}
		self.length_dist = {'H+':[],'H-':[],'M':[]}
		self.amino_freq = {'H+':{'H+':{},'H-':{}},'H-':{'H+':{},'H-':{}},'M':{'H+':{},'H-':{}}}

	def getStats(self, name, path, seq):
		self.compLongestPhobic(name, path)
		self.compBigMixPer(name, path)
		self.compLengthDist(path)
		self.compAminoFreq(path, seq)

	def dump(self):
		print self.longest_phobic
		print self.big_mix_per

		import matplotlib.pyplot as plt
		import numpy as np
		from scipy.stats import gaussian_kde
		data = self.length_dist['H+']
		density = gaussian_kde(data)
		xs = np.linspace(0,8,200)
		density.covariance_factor = lambda : .25
		density._compute_covariance()
		plt.plot(xs,density(xs))
		plt.show()

		print self.length_dist


		print self.amino_freq

	def compLongestPhobic(self, name, path):
		score = 0
		for p in path:
			if p == 'H-':
				score += 1
			else:
				score = 0
		if score > self.longest_phobic['score']:
			self.longest_phobic['name'] = name
			self.longest_phobic['score'] = score

	def compBigMixPer(self, name, path):
		score = 0.0
		for p in path:
			if p == 'M':
				score += 1.0
		score = score/len(path)
		if score > self.big_mix_per['score']:
			self.big_mix_per['score'] = score
			self.big_mix_per['name'] = name

	def compLengthDist(self, path):
		score = 0
		state = path[0]
		for p in path:
			if p != state:
				self.length_dist[state].append(score)
				state = p
				score = 1
			else:
				score += 1
		self.length_dist[state].append(score)

	def compAminoFreq(self, path, seq):
		for i, p in enumerate(path):
			if seq[i] in HYDROPHOBIC:
				if seq[i] not in self.amino_freq[p]['H-']:
					self.amino_freq[p]['H-'][seq[i]] = 1
				else:
					self.amino_freq[p]['H-'][seq[i]] += 1
			else:
				if seq[i] not in self.amino_freq[p]['H+']:
					self.amino_freq[p]['H+'][seq[i]] = 1
				else:
					self.amino_freq[p]['H+'][seq[i]] += 1

class HMM:

	def __init__(self, states, priorP, transitionP, emissionP):
		self.S = states
		self.P = priorP
		self.T = transitionP
		self.E = emissionP

	def viterbi(self, seq):
		# Initialization		
		self.V = {0: {s: self.P[s]*self.E[s][seq[0]] for s in self.S} }
		self.Vp = {0: {s: None for s in self.S} }

		# steps
		for l, x in enumerate(seq):
			if l == 0:
				continue

			self.V[l] = {}
			self.Vp[l] = {}

			for s in self.S:
				maxV = - float("inf")
				maxP = 0
				for p in self.S:
					score = self.V[l-1][p] + math.log(self.T[p][s]) + math.log(self.E[s][x])
					if score > maxV:
						maxV = score
						maxP = p

				self.V[l][s] = maxV
				self.Vp[l][s] = maxP

		# trace back
		path = [0 for i in self.V]
		maxV = - float("inf")
		L = len(self.Vp)-1

		for v in self.V[L]:
			if self.V[L][v] > maxV:
				maxV = self.V[L][v]
				path[L] = v
		
		for l in range(0,L):
			path[L-l-1] = self.Vp[L-l][path[L-l]]

		return path

def hmmFASTA(file):
	s = ['H+', 'H-', 'M']
	p = {'H+' : 1.0/3.0, 'H-' : 1.0/3.0, 'M' : 1.0/3.0}
	t = {'H+' : {
		'H+' : 7.0/8.0, 'H-' : 3.0/80.0, 'M' : 7.0/80.0
	}, 'H-' : {
		'H+' : 1.0/25.0, 'H-' : 4.0/5.0, 'M' : 4.0/25.0
	}, 'M' : {
		'H+' : 1.0/14.0, 'H-' : 1.0/14.0, 'M' : 6.0/7.0
	}}
	e = {'H+' : 
		dict({a : 1.0/40.0 for a in HYDROPHOBIC}.items() + 
		{a : 1.0/15.0 for a in HYDROPHILIC}.items()),
		'H-' : 
		dict({a : 3.0/40.0 for a in HYDROPHOBIC}.items() + 
		{a : 1.0/30.0 for a in HYDROPHILIC}.items()),
		'M' : 
		dict({a : 1.0/20.0 for a in HYDROPHOBIC}.items() + 
		{a : 1.0/20.0 for a in HYDROPHILIC}.items()),
	}
	hmm = HMM(s, p, t, e)
	fasta_sequences = SeqIO.parse(open(file),'fasta')
	statistics = Stats()

	results = {}
	for fasta in fasta_sequences:
		name, sequence = fasta.id, str(fasta.seq)
		print name
		path = hmm.viterbi(sequence)
		print path
		results = {s: path.count(s) for s in set(path)}
		statistics.getStats(name, path, sequence)

	statistics.dump()


# hmmFASTA('test.fa')
if len(sys.argv) > 2:
	hmmFASTA(sys.argv[2])
else:
	hmmFASTA('hw3_proteins.fa')

