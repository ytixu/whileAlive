import operator
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

KEYS = {
    'L_Shoulder': 5,
    'L_Elbow': 6,
    'L_Wrist': 7,
    'R_Shoulder': 2,
    'R_Elbow': 3,
    'R_Wrist': 4,
    'L_Hip': 11,
    # 'L_Knee': 12,
    # 'L_Ankle': 13,
    'R_Hip': 8,
    # 'R_Knee': 9,
    # 'R_Ankle': 10,

    'L_Eye': 15,
    'R_Eye': 14,
    'L_Ear': 17,
    'R_Ear': 16,
    'Nose': 0,

    'Neck': 1
}

sorted_keys = sorted(KEYS.items(), key=operator.itemgetter(1))
sorted_keys = [k for k,_ in sorted_keys]

_W = 720
_H = 480

header = [k+'_x' for k in sorted_keys] + [k+'_y' for k in sorted_keys]
json_pose_dump = {
    'header' : header,
}

flic_poses = json.load(open('flic_poses.json', 'r'))


def distance(pos1, pos2):
    common_keys = set(pos1.keys()).intersection(pos2.keys())
    a = np.array([pos1[k] for k in common_keys])
    b = np.array([pos2[k] for k in common_keys])
    return np.linalg.norm(a-b)

def add_to_img(coords, color, size):
    coords = np.round(coords).astype(int)
    n = len(coords)/2
    plt.scatter(x=coords[:n], y=coords[n:], c=color, s=size)

files = glob.glob('./output/*.json')
for i, json_file in enumerate(files):
    file_id = os.path.basename(json_file).replace('_keypoints.json', '')
    if 'mukbang' in file_id:
        continue
    poses = json.load(open(json_file, 'r'))
    flic_pose = {flic_poses['header'][i]: c for i,c in enumerate(flic_poses[file_id])}

    # img = mpimg.imread('/home/adaptation/yxu219/Documents/FLIC/images/'+ file_id +'.jpg')
    # plt.imshow(img)
    # add_to_img(flic_poses[file_id], 'r', 80)

    best_pose = None
    best_dist = np.inf
    for pose in poses['people']:
        openpose_pose = {header[i]: pose['pose_keypoints'][KEYS[k]*3] for i, k in enumerate(sorted_keys)}
        openpose_pose.update({header[len(sorted_keys)+i]: pose['pose_keypoints'][KEYS[k]*3+1] for i, k in enumerate(sorted_keys)})
        dist = distance(openpose_pose, flic_pose)
        if dist < best_dist:
            best_pose = pose['pose_keypoints']
            best_dist = dist

    if best_dist < np.inf:
        line = [best_pose[KEYS[k]*3] for k in sorted_keys] + [best_pose[KEYS[k]*3+1] for k in sorted_keys]
        json_pose_dump[file_id] = line

        # img = add_to_img(line, 'b', 20)
        # plt.pause(0.05)
        # plt.clf()
    else:
        print 'Not good'
    print best_dist

with open('openpose_pose.json', 'w') as outfile:
    json.dump(json_pose_dump, outfile)
