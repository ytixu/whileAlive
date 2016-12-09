import matplotlib.pyplot as plt
import numpy as np
import random

NUCS = ['A','G','T','C']

substitution = {
  'AG':-1,'GA':-1,
  'TC':-1, 'CT':-1,
  'AC':-2,'CA':-2,
  'GT':-2,'TG':-2,
  'TA':-2,'AT':-2,
  'CG':-2,'GC':-2,
  }
match        =  2

def compute_match_score(ref, qry ,p):
  score = match * p
  for nuc in NUCS:
    if ref == nuc:
      continue
    score += substitution[qry+nuc]*(1-p)/3.0
  return score

def compute_sub_score(ref, qry ,p):
  score = match * (1-p)/3.0 + substitution[qry+ref]*p
  for nuc in NUCS:
    if qry == nuc or ref == nuc:
      continue
    score += substitution[qry+nuc]*(1-p)/3.0
  return score


def match_score(query, ref, prob):
	score = 0
	for i, a in enumerate(query):
		if a == ref[i]:
			score += compute_match_score(ref[i], a, prob[i])
		else:
			score += compute_sub_score(ref[i], a, prob[i])

	return score

def recurse_p(at, stop, acc=[]):
	if at == stop:
		return [acc]

	ps = []
	for p in range(50, 101, 25):
		ps += recurse_p(at+1, stop, acc + [p*0.01])

	return ps

def recurse_n(at, stop, acc=''):
	if at == stop:
		return [acc]

	ns = []
	for n in NUCS:
		ns += recurse_n(at+1, stop, acc+n)

	return ns

def random_seq(l):
	n = ''
	for i in range(l):
		r = random.randint(0,3)
		n += NUCS[r]
	return n


def score_plot(l, allCombo=False):
	dist = [0 for i in range(-match*l, match*l+1)]
	ref = 'A'*l
	ps = recurse_p(0, l)
	ll = len(ps)
	ns = recurse_n(0, l)
	for i, p in enumerate(ps):
		print i, ll
		if allCombo:
			for qry in ns:
				score = int(match_score(qry, ref, p))
				dist[score+l*match] += 1.00
		else:
			for j in range(100):
				qry = random_seq(l)
				score = int(match_score(qry, ref, p))
				dist[score+l*match] += 1.00

	return dist
	
# plt.figure()
# dist = score_plot(7)
# total = sum(dist)
# line1 = plt.plot(range(-match*7, match*7+1), [i/total for i in dist])
# plt.setp(line1, color='#333333', linewidth=4.0)
# dist = score_plot(3, True)
# total = sum(dist)
# line2 = plt.plot(range(-match*3, match*3+1), [i/total for i in dist])
# plt.setp(line2, color='#000000', linewidth=1.0)
# # plt.legend(handles=[line1, line2], loc=1)
# plt.ylabel('Frequency')
# plt.xlabel('Score')
# plt.title('Score Distribution for Word Length 3 and 7')
# plt.show()



probs = open('../chr22.maf.ancestors.42000000.complete.boreo.conf.txt', 'r').readline().strip().split(' ')
dist = {}
total = 0
print len(probs)
for i,c in enumerate(probs):
	p = float(c)
	total += 1
	for j in range(1, 101, 1):
		if p <= j/100.0:
			if j in dist:
				dist[j] += 1

			else:
				dist[j] = 1

		if i == 1006:
			plt.figure()
			keys = dist.keys()
			keys = sorted(keys)
			plt.plot([k/100.0 for k in keys], [dist[k]*1.00/total for k in keys])
			plt.ylabel('Frequency')
			plt.xlabel('Confidence Value')
			plt.title('Cumulative Confidence Distribution in CHR22 (first 1000 bp)')
			plt.show()
			break

