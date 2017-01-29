
# version for numpy 1.8.2
# version for scipy 0.13.3
# version for sklearn 0.18.1
import sys
import math

import numpy as np

from numpy import linalg as LA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model.logistic import LogisticRegression as LogReg
from sklearn.metrics.classification import log_loss
import matplotlib.pyplot as plt


#### problem 1

#### part b
def getCrossValSets(matX, vecY, n=5):
	kf = KFold(n_splits=n)
	for v_i, t_i in kf.split(matX):
		vX, vY, tX, tY = matX[v_i], vecY[v_i], matX[t_i], vecY[t_i]
		yield (vX, vY), (tX, tY)

def runRegression(matX, vecY, C=[1.0], kFold=5):
	acc = [[],[]]
	entropy = [[],[]]
	w_norm = []
	weights = []

	#### part c
	reg = LogReg(penalty='l2')

	#### part b
	# add 1 to each input vector
	matX = np.append(matX, np.array([[1]]*matX.shape[0]), axis=1)
	for train, test in getCrossValSets(matX, vecY):
		for c in C:
			reg.set_params(C=c)
			# TRAINING
			t_score = []
			t_loss = []
			for traing, valid in getCrossValSets(train[0], train[1]):
				reg.fit(traing[0], traing[1])
				# accuracy
				t_score += [reg.score(valid[0], valid[1])]
				# entropy
				predY = reg.predict(valid[0])
				t_loss += [log_loss(valid[1], predY, labels=[0,1])]

			t_score = np.mean(t_score)
			t_loss = np.mean(t_loss)
			# print 'training: ', t_score, t_loss

			# TESTING
			reg.fit(train[0], train[1])
			# accuracy
			v_score = reg.score(test[0], test[1])
			acc[0] += [t_score]
			acc[1] += [v_score]
			# entropy
			predY = reg.predict(test[0])
			v_loss = log_loss(test[1], predY, labels=[0,1])
			entropy[0] += [t_loss]
			entropy[1] += [v_loss]
			# print 'test: ', v_score, v_loss

	for c in C:
		reg.set_params(C=c)
		reg.fit(matX, vecY)
		# norm
		w = reg.coef_[0]
		# print 'norm: ', w, LA.norm(w)
		weights += [w]
		w_norm += [LA.norm(w)]



	return (acc, np.array(weights).transpose().tolist(), w_norm, entropy)

	# print acc
	# print weights
	# print w_norm
	# print entropy

def plotGraph(x, y, labels, loc=2):
	plt.xlabel(labels[0])
	plt.ylabel(labels[1])
	plt.title(labels[2])

	if type(y[0]) == type([]):
		if len(y) == 2:
			plt.plot(x, y[0], label='training', linestyle='--')
			plt.plot(x, y[1], label='test')
			plt.legend(loc=loc)
		else:
			for i,yy in enumerate(y):
				plt.plot(x, yy, label='w'+str(i))

	else:
		plt.plot(x, y)
	plt.show()

# part f
def gaussianBasis(matX, mean, std):
	return np.array(map(lambda vecX :
			map(lambda x : math.exp(-(x-mean)**2/(2.0*std**2))/2.0/math.pi/std, vecX),
		matX))

ld = [1e-10,0.1,1,10.0,100.0,1000.0]
# set a large C for lambda = 0 (no penalty)
ld_n = len(ld)
C = map(lambda x: 1.0/x, ld)

if len(sys.argv) < 3:
	#### test
	matX = np.append(np.random.randint(low=2,high=4,size=(60,2)),
					  np.random.randint(low=20,high=22,size=(40,2)), axis=0)
	vecY = np.append(np.random.randint(2, size=60), np.array([0]*40))
else:
	matX = np.array(map(lambda x: map(float, x.split('\t')), open(sys.argv[2], 'r').read().strip().split('\n')))
	vecY = np.array(map(float, open(sys.argv[3], 'r').read().strip().split('\n')))
# print matX
# print vecY
n = 5

acc, w, wn, e = runRegression(matX, vecY, C, n)
for i in range(ld_n):
	for j in range(ld_n+i, len(acc[0]), ld_n):
		acc[0][i] += acc[0][j]
		acc[1][i] += acc[1][j]
		e[0][i] += e[0][j]
		e[1][i] += e[1][j]

acc = [map(lambda x: x/5.0, acc[0][0:ld_n]), map(lambda x: x/5.0, acc[1][0:ld_n])]
e = [map(lambda x: x/5.0, e[0][0:ld_n]), map(lambda x: x/5.0, e[1][0:ld_n])]

# plots
l = np.log(ld)
# plotGraph(l, acc, ['log(lambda)', 'accuracy', 'Accuracy v.s. Regularization'])
# plotGraph(l, wn, ['log(lambda)', 'weight norm', 'Weight Norms v.s. Regularization'])
# plotGraph(l, w, ['log(lambda)', 'weight', 'Weight v.s. Regularization'])
# plotGraph(l, e, ['log(lambda)', 'entropy', 'Entropy v.s. Regularization'], 3)

# part e
sigma = [0.1, 0.5, 1.0, 5.0, 10.0]
for m in range(-10, 11, 4):
	accuracy, entropy = [[],[]], [[],[]]
	for s in sigma:
		matX = gaussianBasis(matX, m, s)
		acc, w, wn, e = runRegression(matX, vecY, C, n)
		accuracy[0] += [np.mean(acc[0])]
		accuracy[1] += [np.mean(acc[1])]
		entropy[0] += [np.mean(e[0])]
		entropy[1] += [np.mean(e[1])]

	acc = [map(lambda x: x/5.0, accuracy[0]), map(lambda x: x/5.0, accuracy[1])]
	e = [map(lambda x: x/5.0, entropy[0]), map(lambda x: x/5.0, entropy[1])]

	plotGraph(sigma, acc, ['log(lambda)', 'accuracy', 'Accuracy v.s. Regularization mean= '+str(m)])
	plotGraph(sigma, e, ['log(lambda)', 'entropy', 'Entropy v.s. Regularization mean= '+str(m)])

#### Problem 2

def k(matX, z, d):
	return np.array(map(lambda x: (np.dot(x, z) + 1)**d, matX))

def getKernal(matX, d):
	return np.array(map(lambda x: k(matX, x, d), matX))

accuracy, entropy = [[],[]], [[],[]]

ld = [1, 2, 3]
for d in ld:
	newX = getKernal(matX, d)
	acc, w, wn, e = runRegression(newX, vecY, [1e10], n)
	accuracy[0] += [np.mean(acc[0])]
	accuracy[1] += [np.mean(acc[1])]
	entropy[0] += [np.mean(e[0])]
	entropy[1] += [np.mean(e[1])]

acc = [map(lambda x: x/5.0, accuracy[0]), map(lambda x: x/5.0, accuracy[1])]
e = [map(lambda x: x/5.0, entropy[0]), map(lambda x: x/5.0, entropy[1])]

plotGraph(ld, acc, ['log(lambda)', 'accuracy', 'Accuracy v.s. Regularization'])
plotGraph(ld, e, ['log(lambda)', 'entropy', 'Entropy v.s. Regularization'])
