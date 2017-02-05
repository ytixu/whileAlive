
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
	kf = KFold(n_splits=n, shuffle=True)
	for v_i, t_i in kf.split(matX):
		vX, vY, tX, tY = matX[v_i], vecY[v_i], matX[t_i], vecY[t_i]
		return (vX, vY), (tX, tY)

def homogeneousCoord(matX):
	return np.append(matX, np.array([[1]]*matX.shape[0]), axis=1)

def runRegression(matX, vecY, C=[1e10], kFold=5, kernalized=0, d=0, splitted=False):
	acc = [[],[]]
	entropy = [[],[]]
	w_norm = []
	weights = []

	#### part c
	reg = LogReg(penalty='l2')
	if splitted:
		trainMat = matX
		testMat = vecY

	else:
		#### part b
		trainMat, testMat = getCrossValSets(matX, vecY)
	# add 1 to each input vector
	if kernalized:
		#### for kernel
		test = (homogeneousCoord(computeSimilarity(trainMat[0], testMat[0], d)), testMat[1])
		train = (homogeneousCoord(getKernal(trainMat[0], d)), trainMat[1])
	else:
		train = (homogeneousCoord(trainMat[0]), trainMat[1])
		test = (homogeneousCoord(testMat[0]), testMat[1])

	for c in C:
		reg.set_params(C=c)
		# TRAINING
		reg.fit(train[0], train[1])
		# accuracy
		acc[0] += [reg.score(train[0], train[1])]
		acc[1] += [reg.score(test[0], test[1])]
		# entropy
		predY = reg.predict(train[0])
		entropy[0] += [log_loss(train[1], predY, labels=[0,1])]
		predY = reg.predict(test[0])
		entropy[1] += [log_loss(test[1], predY, labels=[0,1])]
		# norm
		w = reg.coef_[0]
		# print 'norm: ', len(w), LA.norm(w)
		weights += [w]
		w_norm += [LA.norm(w)]

	return (acc, np.array(weights).transpose().tolist(), w_norm, entropy, trainMat, testMat)

	# print acc
	# print weights
	# print w_norm
	# print entropy

def plotGraph(x, y, labels, loc=2, save=None):
	fg = plt.figure()
	plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
	plt.xlabel(labels[0])
	plt.ylabel(labels[1])
	plt.title(labels[2])

	if type(y[0]) == type([]):
		if len(y) < 5:
			plt.plot(x, y[0], label='training', linestyle='--')
			plt.plot(x, y[1], label='test')
			if len(y) == 4:
				plt.plot(x, y[2], label='previous training', linestyle='--')
				plt.plot(x, y[3], label='previous test')
			plt.legend(loc=loc)
		else:
			for i,yy in enumerate(y):
				plt.plot(x, yy, label='w'+str(i))

	else:
		plt.plot(x, y)

	#########	UNCOMMENT THIS TO SAVE
	if save:
		fg.savefig(save, dpi=fg.dpi)

	plt.show()

# part e and g
def gaussianBasis(matX, means, stds):
	if type(stds) != type([]):
		stds = [stds]
	newX = []
	for i,vecX in enumerate(matX):
		newX += [[]]
		for x in vecX:
			for std in stds:
				for m in means:
					newX[i] += [math.exp(-(x-m)**2/(2.0*std**2))/math.sqrt(2.0*math.pi)/std]

	return np.array(newX)

ld = [1e-10,0.1,1,10.0,100.0,1000.0,10000.0]
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

n = 5
acc, w, wn, e, trainMat, testMat = runRegression(matX, vecY, C, n)

# plots for d
print "q1, d"
l = np.log10(ld)
plotGraph(l, acc, ['log(lambda)', 'accuracy', 'Accuracy v.s. Regularization'], save="q1-d-acc.png")
plotGraph(l, wn, ['log(lambda)', 'weight norm', 'Weight Norms v.s. Regularization'], save="q1-d-norm.png")
plotGraph(l, w, ['log(lambda)', 'weight', 'Weight v.s. Regularization'], save="q1-d-w.png")
plotGraph(l, e, ['log(lambda)', 'entropy', 'Entropy v.s. Regularization'], 3, save="q1-d-e.png")

# part f
sigmas = [0.1, 0.5, 1.0, 5.0, 10.0]
means = range(-10, 11, 5)
acc = [[],[], [acc[0][0]]*len(sigmas), [acc[1][0]]*len(sigmas)]
e = [[],[], [e[0][0]]*len(sigmas), [e[1][0]]*len(sigmas)]

for s in sigmas:
	newTrainMat = (gaussianBasis(trainMat[0], means, s,), trainMat[1])
	newTestMat = (gaussianBasis(testMat[0], means, s,), testMat[1])
	acc_g, _, _, e_g, _, _ = runRegression(newTrainMat, newTestMat, splitted=True)
	acc[0] += acc_g[0]
	acc[1] += acc_g[1]
	e[0] += e_g[0]
	e[1] += e_g[1]

print "q1, f"
plotGraph(sigmas, acc, ['sigma', 'accuracy', 'Accuracy v.s. STD'], save="q1-f-acc.png")
plotGraph(sigmas, e, ['sigma', 'entropy', 'Entropy v.s. STD'], save="q1-f-e.png")

# part g
accuracy, entropy = [[],[]], [[],[]]
newTrainMat = (gaussianBasis(trainMat[0], means, sigmas,), trainMat[1])
newTestMat = (gaussianBasis(testMat[0], means, sigmas,), testMat[1])
acc, w, wn, e, _, _  = runRegression(newTrainMat, newTestMat, C, splitted=True)
n = len(matX[0])*len(means)
w = np.array(w).transpose().tolist()
weights = map(lambda x: map(lambda i: LA.norm(w[x][n*i:n*(i+1)]), range(len(sigmas))), range(len(w)))
w = np.array(weights).transpose().tolist()

print "q1, g"
plotGraph(l, acc, ['log(lambda)', 'accuracy', 'Accuracy v.s. Regularization'], save="q1-g-acc.png")
plotGraph(l, wn, ['log(lambda)', 'weight norm', 'Weight Norms v.s. Regularization'], save="q1-g-norm.png")
plotGraph(l, w, ['log(lambda)', 'weight', 'Weight v.s. Regularization'], save="q1-g-w.png")
plotGraph(l, e, ['log(lambda)', 'entropy', 'Entropy v.s. Regularization'], save="q1-g-e.png")

### Problem 2

# compte similarity of vector z with all vectors in matX
# for K is kenerl, this outputs a vector [K(x1, z), K(x2, z), ..., K(xn, z)]
def k(matX, z, d):
	return np.array(map(lambda x: (np.dot(x, z) + 1)**d, matX))

# compute similarity between matrix
# for K is kernel, this outputs a matrix
# [[K(x1, z1), K(x2, z1), ..., K(xn, z1)],
# [K(x1, z2), K(x2, z2), ..., K(xn, z2)],
# ...
# [K(x1, zm), K(x2, zm), ..., K(xn, zm)]]
def computeSimilarity(X, Z, d):
	return np.array(map(lambda z: k(X, z, d), Z))

# compute kernel matrix
def getKernal(matX, d):
	return np.array(map(lambda x: k(matX, x, d), matX))

accuracy, entropy = [[],[]], [[],[]]

ld = [1, 2, 3]
for d in ld:
	acc, w, wn, e, _, _  = runRegression(matX, vecY, kernalized=1, d=d)
	accuracy[0] += acc[0]
	accuracy[1] += acc[1]
	entropy[0] += e[0]
	entropy[1] += e[1]

print "q2"
plotGraph(ld, accuracy, ['d', 'accuracy', 'Accuracy v.s. d'], save="q2-acc.png")
plotGraph(ld, entropy, ['d', 'entropy', 'Entropy v.s. d'], save="q2-e.png")