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

		
		for l, x in enumerate(seq):
			if l == 0:
				continue

			self.V[l] = {}
			self.Vp[l] = {}

			for s in self.S:
				maxV = 0
				maxP = 0
				for p in self.S:
					score = self.V[p][l]*self.T[p][s]*self.E[s][x]
					if score > maxV:
						maxV = score
						maxP = p

				self.V[l][s] = maxV
				self.Vp[l][s] = maxP

