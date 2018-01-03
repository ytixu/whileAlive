import csv
import json
import numpy as np
import cv2
from scipy.spatial.distance import cosine as scipy_cosine
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

kinova_arm_file = 'random_poses.csv'
open_pose_file = 'openpose_pose.json'


def generate_and_plot_3D(coords, projects):
	xs = [coords[i] for i in range(0, len(coords), 3)]
	ys = [coords[i] for i in range(1, len(coords), 3)]
	zs = [coords[i] for i in range(2, len(coords), 3)]

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot(xs, ys, zs)

	xs = [x[0][0] for x in projects]
	ys = [1 for i in range(len(projects))]
	zs = [x[0][1] for x in projects]

	ax.plot(xs, ys, zs)

	plt.show()


def project(coords, y):
	rvec = np.identity(3)
	tvec = np.array([0.0,y,2.0])
	cameraMatrix = np.identity(3)
	cameraMatrix[0][0] = 1
	cameraMatrix[1][1] = 0.2
	cameraMatrix[:,2] = 0
	points = np.zeros((len(coords)/3, 3))
	for i in range(3):
		points[:,i] = [coords[j] for j in range(i, len(coords), 3)]
	img_points, _ = cv2.projectPoints(points, rvec, tvec, cameraMatrix, None)
	img_points[:,:,0] = img_points[:,:,0]-img_points[0][0][0]
	img_points[:,:,1] = img_points[:,:,1]-img_points[0][0][1]
	return img_points


def plot(rpose, hpose):
	x = rpose[:len(rpose)/2]
	y = rpose[len(rpose)/2:]
	plt.plot(x, y, color='r')

	x = [hpose[i] for i in range(0,len(hpose),2)]
	y = [hpose[i] for i in range(1,len(hpose),2)]
	plt.plot(x, y, color='b')
	plt.show()


# line = [-1,-1,-1, 1,-1,-1, 1,-1,1, -1,-1,1]
# y = -4
# p = project(line, y)
# generate_and_plot_3D(line, p)


def compare(rpose, hpose, r_idx = [7,13], h_idx=[6,7]):
	return scipy_cosine(rpose[r_idx]/np.linalg.norm(rpose[r_idx]), hpose[h_idx]/np.linalg.norm(hpose[h_idx]))

def normalize(h_pose):
	h_pose = [h_pose[i*2+j] - h_pose[j] for i in range(len(h_pose)/2) for j in range(2)]
	n = np.linalg.norm([h_pose[2], h_pose[3]])
	return [-h_pose[i*2+j] / n for i in range(len(h_pose)/2) for j in range(2)]

header_idx = []

robot_coords = {}

with open(kinova_arm_file, 'r') as csvfile:
	spamreader = csv.reader(csvfile, delimiter='\t')
	for _id, row in enumerate(spamreader):
		if _id == 0:
			header_idx = [(i,k) for i,k in enumerate(row) if ('j2s7s300::' in k) and ('-o' not in k)]
			# print len(header_idx)
		else:
			# print row
			coords = [float(row[i]) for i,_ in header_idx]
			y = -10
			p = project(coords, y)
			robot_coords[_id] = (np.concatenate((p[:,0,0],p[:,0,1])), p, coords)
			# print p
			# generate_and_plot_3D(coords, p)

# print robot_coords

open_poses = json.load(open(open_pose_file, 'r'))
hip_idx = [i for i in range(len(open_poses['header'])) if 'R_Hip' in open_poses['header'][i]]
shoulder_idx = [i for i in range(len(open_poses['header'])) if 'R_Shoulder' in open_poses['header'][i]]
wrist_idx = [i for i in range(len(open_poses['header'])) if 'R_Wrist' in open_poses['header'][i]]
elbow_idx = [i for i in range(len(open_poses['header'])) if 'R_Elbow' in open_poses['header'][i]]
human_coords = {k:	[p[i] for i in hip_idx] +
					[p[i] for i in shoulder_idx] +
					[p[i] for i in elbow_idx] +
					[p[i] for i in wrist_idx] for k,p in open_poses.iteritems() if k != 'header'}
human_coords = {k: normalize(p) for k,p in human_coords.iteritems()}


for _id, hp_raw in human_coords.iteritems():
	s_best = np.inf
	k_best = None
	hp = np.array(hp_raw)
	# plot(hp, human_coords_raw[_id])
	for k, rp_raw in robot_coords.iteritems():
		rp, _, _ = rp_raw
		s_new = compare(rp, hp)
		# print s_new
		if s_new < s_best:
			s_best = s_new
			k_best = k

	rp, rp_conv, rp_coords = robot_coords[k_best]
	plot(rp, hp)
	generate_and_plot_3D(rp_coords, rp_conv)
