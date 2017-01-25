
# version for numpy 1.8.2
# version for scipy 0.13.3
# version for sklearn 0.18.1
import sys

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
		# for tv_i, vv_i in kf.split(vX):
		# 	# training, validation and test sets
		# 	yield (vX[tv_i], vY[tv_i]), (vX[vv_i], vY[vv_i]), (tX, tY)

def runRegression(matX, vecY, C=[1.0], kFold=5):
	acc = [[],[]]
	entropy = []
	w_norm = []
	weights = [[],[],[]]

	#### part c
	reg = LogReg(penalty='l2')

	#### part b
	# add 1 to each input vector
	matX = np.append(matX, np.array([[1]]*matX.shape[0]), axis=1)
	for train, test in getCrossValSets(matX, vecY):
		for c in C:
			reg.set_params(C=c)

			# accuracy
			scores = cross_val_score(reg, train[0], train[1], cv=kFold, verbose=1)
			t_score = np.mean(scores)
			# print 'training: ', scores, t_score
			reg.fit(train[0], train[1])
			v_score = reg.score(test[0], test[1])
			# print 'test: ', v_score
			acc[0] += [t_score]
			acc[1] += [v_score]

			# norm
			w = reg.coef_[0]
			# print 'norm: ', w, LA.norm(w)
			weights[0] += [w[0]]
			weights[1] += [w[1]]
			weights[2] += [w[2]]
			w_norm += [LA.norm(w)]

			# cross entropy
			predY = reg.predict(matX)
			loss = log_loss(vecY, predY, labels=[0,1])
			# print 'loss: ', loss
			entropy += [loss]

	return (acc, weights, w_norm, entropy)

	# print acc
	# print weights
	# print w_norm
	# print entropy

def plotGraph(x, y, n, labels):
	plt.xlabel(labels[0])
	plt.ylabel(labels[1])
	plt.title(labels[2])
	for i in range(n):
		plt.plot(x, y[i*len(y)/n:(i+1)*len(y)/n])
	plt.show()


l = [1e-10,0.1,1,10.0,100.0,1000.0]
# set a large C for lambda = 0 (no penalty)
C = map(lambda x: 1.0/x, l)

#### test
testX = np.append(np.random.randint(low=2,high=4,size=(60,2)),
				  np.random.randint(low=20,high=22,size=(40,2)), axis=0)
testY = np.append(np.random.randint(2, size=60), np.array([0]*40))
# print testX
# print testY
n = 5
acc, w, wn, e = runRegression(testX, testY, C, n)

# plots
l = np.log(l)
plotGraph(l, acc[0], n, ['log(lambda)', 'accuracy', 'Training Accuracy v.s. Regularization'])
plotGraph(l, acc[1], n, ['log(lambda)', 'accuracy', 'Test Accuracy v.s. Regularization'])
plotGraph(l, wn, n, ['log(lambda)', 'weight norm', 'Weight Norms v.s. Regularization'])
plotGraph(l, w[0], n, ['log(lambda)', 'weight', 'Weight v.s. Regularization'])
plotGraph(l, w[1], n, ['log(lambda)', 'weight', 'Weight v.s. Regularization'])
plotGraph(l, w[2], n, ['log(lambda)', 'weight', 'Weight v.s. Regularization'])
plotGraph(l, e, n, ['log(lambda)', 'entropy', 'Entropy v.s. Regularization'])


