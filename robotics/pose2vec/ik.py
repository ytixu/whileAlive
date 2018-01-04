import ikpy
import numpy as np


def ik_poses():
	arm_chain = ikpy.chain.Chain.from_urdf_file('./j2s7s300.urdf', base_elements=['j2s7s300_link_base'])
	some_pose = arm_chain.forward_kinematics(np.array([0.0,0.0,2.9,0.0,1.3,4.2,1.4,0.0,0.0]))
	# target_vector = [0.5,0.5,0.5]
	# target_frame = np.eye(4)
	# target_frame[:3, 3] = target_vector
	print some_pose, arm_chain.inverse_kinematics(some_pose)

ik_poses()