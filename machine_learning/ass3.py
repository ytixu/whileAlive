import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
import csv

X = None
R = None
D = None
d = 0

with open('hw3pca.txt', 'rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
	x = []
	for row in spamreader:
		x += [map(lambda x: float(x.strip()), row[:-1])]
		d = len(row)

	X = np.array(x)
	D = range(2,d,5)

# X = X[:20]
k = 5 # k-fold cross validation
kf = KFold(len(X), n_folds=5, shuffle=True)
R_test = [0 for n in D]
R_train = [0 for n in D]
Var = [0 for n in D]

for train_index, test_index in kf:
	X_train, X_test = X[train_index], X[test_index]
	print ':)'

	for n in D:
		index = (n-2)/k
		pca = PCA(n_components=n)
		pca.fit(X_train)
		print pca.components_
		Var[index] += pca.explained_variance_ratio_[0]/k
		r = 0
		for x in X_train:
			xx = pca.inverse_transform(pca.transform(x))
			r += np.linalg.norm((x-xx), ord=2)**2
		R_train[index] += r/k
		r = 0
		for x in X_test:
			xx = pca.inverse_transform(pca.transform(x))
			r += np.linalg.norm((x-xx), ord=2)**2
		R_test[index] += r/k


# print R.values()
import matplotlib.pyplot as plt
plt.plot(D, R_test)
plt.plot(D, R_train)
plt.legend(['test', 'train'], loc='lower left')
plt.ylabel('log(Reconstruction Error)')
plt.xlabel('Dimension')
plt.yscale('log')
plt.show()

plt.plot(D, Var)
plt.xlabel('Dimension')
plt.ylabel('Variance of top Eigen Value')
plt.show()
