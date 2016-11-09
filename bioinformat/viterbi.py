from Bio import SeqIO ### this is only for reading FASTA file
import math

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
		'H+' : 19.0/20.0, 'H-' : 3.0/200.0, 'M' : 7.0/200.0
	}, 'H-' : {
		'H+' : 1.0/50.0, 'H-' : 9.0/10.0, 'M' : 2.0/25.0
	}, 'M' : {
		'H+' : 1.0/30.0, 'H-' : 1.0/30.0, 'M' : 14.0/15.0
	}}
	e = {'H+' : 
		dict({a : 0.2/8.0 for a in ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']}.items() + 
		{a : 0.8/12.0 for a in ['R', 'H', 'K', 'D', 'E', 'T', 'N', 'Q', 'S', 'C', 'G', 'P']}.items()),
		'H-' : 
		dict({a : 0.6/8.0 for a in ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']}.items() + 
		{a : 0.4/12.0 for a in ['R', 'H', 'K', 'D', 'E', 'T', 'N', 'Q', 'S', 'C', 'G', 'P']}.items()),
		'M' : 
		dict({a : 0.05 for a in ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']}.items() + 
		{a : 0.05 for a in ['R', 'H', 'K', 'D', 'E', 'T', 'N', 'Q', 'S', 'C', 'G', 'P']}.items()),
	}
	hmm = HMM(s, p, t, e)
	fasta_sequences = SeqIO.parse(open(file),'fasta')
	for fasta in fasta_sequences:
		name, sequence = fasta.id, str(fasta.seq)
		print name
		path = hmm.viterbi(sequence)
		print {s: path.count(s) for s in set(path)}


# hmmFASTA('test.fa')
hmmFASTA('hw3_proteins.fa')

