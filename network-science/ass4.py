#https://github.com/kjahan/community/blob/master/cmty.py

import pickle
import community	
import networkx as nx
import matplotlib.pyplot as plt

from collections import Counter, defaultdict
import random


MAX_DIFF = 0.01
FLOAT_KEYS = ['SellingPriceUSD','msrp','accounting_cost_usd']
MUST_MATCH = ['Category1', 'Material', 'Style']

def groups(many_to_one):
	one_to_many = defaultdict(set)
	for v, k in many_to_one.items():
		one_to_many[k].add(v)
	return dict(one_to_many)

def asyn_lpa_communities(G, weight=None):

	labels = {n: i for i, n in enumerate(G)}
	cont = True
	counter = 0
	while cont:
		counter += 1
		# print counter
		if counter > 500:
			break
		cont = False
		nodes = list(G)
		random.shuffle(nodes)
		# Calculate the label for each node
		for node in nodes:
			if len(G[node]) < 1:
				continue

			# Get label frequencies. Depending on the order they are processed
			# in some nodes with be in t and others in t-1, making the
			# algorithm asynchronous.
			label_freq = Counter({labels[v]: G.edge[v][node][weight]
								  if weight else 1 for v in G[node]})

			# Choose the label with the highest frecuency. If more than 1 label
			# has the highest frecuency choose one randomly.
			max_freq = max(label_freq.values())
			best_labels = [label for label, freq in label_freq.items()
						   if freq == max_freq]
			new_label = random.choice(best_labels)
			labels[node] = new_label
			# Continue until all nodes have a label that is better than other
			# neighbour labels (only one label has max_freq for each node).
			cont = cont or len(best_labels) > 1

	# TODO In Python 3.3 or later, this should be `yield from ...`.
	return iter(groups(labels).values())


def percent_diff(a, b):
	return 2*abs(a-b)/(a+b)

def not_null(val):
	if not val or val == 'NA' or val == '0.00' or val == '0':
		return False
	return True

def compute_score(a, b, attr):
	score = 0
	for key in MUST_MATCH:
		if attr[a][key] != attr[b][key]:
			return 0

	for key, val in attr[a].iteritems():
		if key in attr[b]:
			if key not in FLOAT_KEYS:
				if attr[b][key] == val:
					score += 1
			else:
				if percent_diff(float(val), float(attr[b][key])) < MAX_DIFF:
					score += 1
	return score

def getGraph(file_name, save_name):
	import csv

	G = nx.Graph()
	properties = {}
	node_attr = {}
	node_labels = {}
	edge_labels = {}

	with open(file_name, 'r') as csv_file:
		reader = csv.reader(csv_file, delimiter=',', quotechar='"')
		keys = []
		for i, row in enumerate(reader):
			print row[1]
			if i == 0:
				keys = row[1:]
				continue
			prod = row[0]
			node_attr[prod] = {}
			node_labels[prod] = row[1]
			G.add_node(prod, label=row[1])
			sim = []
			for j, val in enumerate(row[1:]):
				key = keys[j]
				if not_null(val):
					if key not in FLOAT_KEYS:
						prop_key = key+'-'+val
						if key in MUST_MATCH:
							if prop_key not in properties:
								properties[prop_key] = []
							else:
								sim = set(list(sim) + properties[prop_key])
							properties[prop_key].append(prod)

					node_attr[prod][key] = val

			for sim_node in sim:
				w = compute_score(sim_node, prod, node_attr)
				if w == 0:
					continue
				G.add_edge(sim_node, prod, weight=w)

			# if i > 40:
			# 	break

	# edge_labels = dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])
	# print edge_labels
	# nx.draw(G, pos=nx.spring_layout(G), labels=node_labels)
	# plt.show()


	pickle.dump(G, open(save_name, 'w'))

def loadGraph(file_name):
	return pickle.load(open(file_name))

def labelPropagation(G):
	communities = asyn_lpa_communities(G)
	partition = {node: com for com, group in enumerate(communities) for node in group}
	print partition
	nx.set_node_attributes(G, 'community', partition)
	
	nx.write_gexf(G, "label-product-lv-cat.gexf")

def optimization(G):
	partition = community.best_partition(G)
	# size = float(len(set(partition.values())))
	# pos = nx.random_layout(G)
	# count = 0.
	nx.set_node_attributes(G, 'community', partition)
	to_draw = [[key for key, val in partition.iteritems() if val == com] for com in set(partition.values())]
	for count, list_nodes in enumerate(to_draw):
		print count
		print list_nodes

	nx.write_gexf(G, "opt-product-lv-cat.gexf")
	# drawGraph(G, to_draw)

def drawGraph(G,partitions):
	size = float(len(partitions))-1
	pos = nx.spectral_layout(G)
	for count, list_nodes in enumerate(partitions):
		nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
									node_color = str(count*1.0 / size))

	nx.draw_networkx_edges(G,pos, alpha=0.5)
	plt.show()

# getGraph('product-lv-cat1.csv', 'graph-cat.txt')
G = loadGraph('graph-cat.txt')
# print 'label prob'
labelPropagation(G)
# print ''
# print 'opt'
# optimization(G)

