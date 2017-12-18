import csv
import json

kinova_arm_file = 'random_poses.csv'
open_pose_file = 'openpose_pose.json'

kinova_arm_header = None
with open(kinova_arm_file, 'r') as csvfile:
	spamreader = csv.reader(csvfile, delimiter='\t')
	for i,row in enumerate(spamreader):
		if i == 0:
			kinova_arm_header = row
			print kinova_arm_header


open_poses = json.load(open(open_pose_file, 'r'))

