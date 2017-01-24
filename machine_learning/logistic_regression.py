import numpy as np
from sklearn.model_selection import KFold
import sklearn.linear_model.LogisticRegression as LogReg

#### problem 1

#### test
testX = [[1,2],[2,2],[2,3],[5,6],[5,5]]
testY =[0,0,0,1,1]

#### part b
def getCrossValSets(matX, vecY, n=5):
	kf = KFold(n_splits=n)
	# add 1 to each input vector
	matX = map(lambda x: x+[1], matX)

	for v_i, t_i in kf.split(matX):
		vX, vY, tX, tY = matX[v_i], vecY[v_i], matX[t_i], vecY[t_i]
		for tv_i, vv_i in kf.split(vX):
			# training, validation and test sets
			yield (vX[tv_i], vY[tv_i]), (vX[vv_i], vY[vv_i]), (tX, tY)

def runRegression(matX, vecY, C=[1.0]):
	
	#### part c
	reg = LogReg(penalty='l2')
	print reg.get_params().keys()

	#### part b
	# for train, val, test in getCrossValSets(matX, vecY):
	# 	for c in C:




C = map(lambda x: 1/x, [1e-320,0.1,1,10,100,1000])

runRegression(testX, testY, C)
