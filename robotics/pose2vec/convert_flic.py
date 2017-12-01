import scipy.io
import json
import operator

KEYS = {
	'L_Shoulder': 1,
	'L_Elbow': 2,
	'L_Wrist': 3,
	'R_Shoulder': 4,
	'R_Elbow': 5,
	'R_Wrist': 6,
	'L_Hip': 7,
	# 'L_Knee': 8,
	# 'L_Ankle': 9,
	'R_Hip': 10,
	# 'R_Knee': 11,
	# 'R_Ankle': 12,

	'L_Eye': 13,
	'R_Eye': 14,
	# 'L_Ear': 15,
	# 'R_Ear': 16,
	'Nose': 17,

	# 'M_Shoulder': 18,
	# 'M_Hip': 19,
	# 'M_Ear': 20,
	# 'M_Torso': 21,
	# 'M_LUpperArm': 22,
	# 'M_RUpperArm': 23,
	# 'M_LLowerArm': 24,
	# 'M_RLowerArm': 25,
	# 'M_LUpperLeg': 26,
	# 'M_RUpperLeg': 27,
	# 'M_LLowerLeg': 28,
	# 'M_RLowerLeg': 29,
}

sorted_keys = sorted(KEYS.items(), key=operator.itemgetter(1))
sorted_keys = [k for k,_ in sorted_keys]

_W = 720
_H = 480

json_pose_dump = {
	'header' : [k+'_x' for k in sorted_keys] + [k+'_y' for k in sorted_keys],
}

mat = scipy.io.loadmat('examples.mat')

for i, pose in enumerate(mat['examples'][0]):
	line = [pose['coords'][0,KEYS[k]-1] for k in sorted_keys]
	line = line + [pose['coords'][1,KEYS[k]-1] for k in sorted_keys]
	json_pose_dump[pose['filepath'][0].replace('.jpg', '')] = line
	print i

with open('/media/yxu219/TOSHIBA EXT/robot/openpose/flic_poses.json', 'w') as outfile:
    json.dump(json_pose_dump, outfile)
