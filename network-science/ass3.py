import csv
import operator
import networkx as nx
from sklearn.preprocessing import scale

def getGraph(brand, centl):
	G = nx.DiGraph()
	w_norm = []
	p_norm = []
	keys = []

	with open('path-data-'+brand+'.csv', 'r') as csv_file:
		reader = csv.reader(csv_file, delimiter=',', quotechar='"')
		w_norm = scale(map(lambda x: float(x[0]), reader), axis=0, with_mean=True, with_std=True, copy=True )
		csv_file.seek(0)
		p_norm = scale(map(lambda x: float(x[0])*float(x[5]), reader), axis=0, with_mean=True, with_std=True, copy=True )

		csv_file.seek(0)
		for i, row in enumerate(reader):

			p = (p_norm[i]+1)/2
			w = (w_norm[i]+1)/2

			G.add_edge(row[1], row[2], weight=w, 
					price=p, 
					rev_weight=1/w if w > 0 else 0,
					rev_price=1/p if p > 0 else 0)

	print brand[0]
	for w in ['rev_weight', 'rev_price']:
		keys.append('closeness-'+brand[0]+'-'+w)
		for node, cln in nx.closeness_centrality(G, distance=w).iteritems():
			if not node in centl:
				centl[node] = {
				 	'closeness-'+brand[0]+'-'+w : cln
				}
			else:
				centl[node]['closeness-'+brand[0]+'-'+w] = cln

	for w in ['weight', 'price']:
		keys.append('betweenness-'+brand[0]+'-'+w)
		keys.append('eigen-'+brand[0]+'-'+w)
		for node, btn in nx.betweenness_centrality(G, weight=w).iteritems():
			centl[node]['betweenness-'+brand[0]+'-'+w] = btn

		for node, eigen in nx.eigenvector_centrality(G, weight=w).iteritems():
			centl[node]['eigen-'+brand[0]+'-'+w] = eigen

	return keys

centl = {}
keys = getGraph('lv', centl)
keys = keys + getGraph('hermes', centl)
keys = keys + getGraph('coach', centl)
keys = keys + getGraph('all', centl)

with open('centrality-result-all.csv', 'w') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"')
	keys = sorted(keys)
	spamwriter.writerow(['ID'] + keys)
	for node, cln in centl.iteritems():
		spamwriter.writerow([node] + [cln[i] if i in cln else 0 for i in keys])

