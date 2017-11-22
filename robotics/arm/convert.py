import csv
import numpy as np

filename = 'data5.csv'
links =  ['j2n6s300_joint_1', 'j2n6s300_joint_2', 'j2n6s300_joint_3', 'j2n6s300_joint_4', 'j2n6s300_joint_5', 'j2n6s300_joint_6', 'j2n6s300_joint_finger_1', 'j2n6s300_joint_finger_2', 'j2n6s300_joint_finger_3']
cart_pos = ['j2n6s300::j2n6s300_link_6-x','j2n6s300::j2n6s300_link_6-y','j2n6s300::j2n6s300_link_6-z','j2n6s300::j2n6s300_link_6-ox','j2n6s300::j2n6s300_link_6-oy','j2n6s300::j2n6s300_link_6oz', 'j2n6s300_joint_finger_1', 'j2n6s300_joint_finger_2', 'j2n6s300_joint_finger_3']
coding = 'abcdefghi'


text = ''
indices = None
prev = None
NORM = np.pi*2
norms = []
n_count = 0
BINS = 9

prev_suffix = 'O'
suffix = 'O'

def get_norms(row_ele, i):
	global indices, norms
	if i == 0:
		indices = [j for j,header in enumerate(row_ele) if header in links[:-3]]
		norms = [[0,10] for i in links]
		norms[-3:] = [[BINS,0]]*3
		return

	for i, idx in enumerate(indices):
		norms[i][0] = max(norms[i][0], float(row_ele[idx])% NORM)
		norms[i][1] = min(norms[i][1], float(row_ele[idx])% NORM)


def abs_pos(row_ele, i):
	global text, prev, prev_suffix, suffix, indices, norm
	if i == 0:
		indices = [j for j,header in enumerate(row_ele) if header in links]
		return

	word = []
	for i, idx in enumerate(indices):
		word.append(str(int(round((float(row_ele[idx]) % NORM - norms[i][1])/norms[i][0]*BINS))).replace('-', 'n')+coding[i])

	# print word
	# print [float(row_ele[i]) % NORM for i in indices]
	word = ''.join(word).replace('f0g0h0i','O').replace('f1g1h1i', 'C')
	if word == prev:
		return

	prev = word

	if word[-1] != suffix:
		if word[-1] == prev_suffix:
			prev_suffix = suffix
			suffix = word[-1]
			word = 'F' + ' ' + word
		else:
			prev_suffix = suffix
			suffix = word[-1]
			word = suffix + ' ' + word
	else:
		prev_suffix = suffix


	text = text + word + ' '

def cat_pos(row_ele, i):
	global text, prev, prev_suffix, suffix, indices

	if i == 0:
		indices = {j:(False if header in links[:3] else True) for j,header in enumerate(row_ele) if header in links}
		return

	joint_pos = []
	for idx, normalize in indices.iteritems():
		if normalize:
			joint_pos.append(float(row_ele[idx]) % NORM)
		else:
			joint_pos.append(float(row_ele[idx]))

	if prev is None:
		prev = joint_pos[:-3]
		return

	word = [str(int(round((joint_pos[i] - prev[i])*10))).replace('-', 'n')+coding[i]for i in range(len(prev))]
	# print joint_pos, word
	word = (''.join(word) + ''.join([str(int(round(i))) for i in joint_pos[-3:]])).replace('f000','O').replace('f111', 'C')
	text = text + word + ' '
	prev = joint_pos[:-3]



with open(filename, 'r') as csvfile:
	spamreader = csv.reader(csvfile)
	for i, row in enumerate(spamreader):
		row_ele = row[0].split('\t')
		get_norms(row_ele, i)

	norms = [(n[0] - n[1], n[1]) for n in norms]
	# print norms

	csvfile.seek(0)
	for i, row in enumerate(spamreader):
		row_ele = row[0].split('\t')
		cat_pos(row_ele, i)
		# if i == 50:
		# 	break


print text