import math
import json
import numpy as np
#TODO: eventually move all helper functions here

with open('ten_strokes.json', 'r') as file:
    strokes = json.load(file)

list1 = strokes[0][0]
list2 = strokes[1][1]


def get_knee_lengths(hip,knee,ankle):
    a = abs(math.dist(knee,ankle))
    b = abs(math.dist(ankle,hip))
    c = abs(math.dist(knee,hip))
    return a,b,c

def get_body_lengths(hip,shoulder):
    a = abs(math.dist(shoulder,hip))
    b = shoulder[1]-hip[1]
    return a,b

def find_angle(a,b,c):
    # finds angle using law of cosines
    right_side = (a**2 - b**2 + c**2)/(2*a*c)
    return math.acos(right_side)


def calc_angles(hip,knee,ankle,shoulder):
    a,b,c = get_knee_lengths(hip=hip,knee=knee,ankle=ankle)
    d,e = get_body_lengths(hip=hip,shoulder=shoulder)
    knee_angle = round(find_angle(a,b,c)*(180/math.pi),1)
    if shoulder[0] > hip[0]:
        body_direction = 1
    elif shoulder[0] == hip[0]:
        body_direction = 0
    else:
        body_direction = -1
    body_angle = round((180-round(math.acos(e / d) * (180 / math.pi), 3)),1)*body_direction
    return knee_angle,body_angle

# [12,14,16,18,24,26,28]

def normalize_t(y_old):
    x_old = np.linspace(0, 1, len(y_old))
    x_new = np.linspace(0, 1, 100)
    return np.interp(x_new, x_old, y_old)

def process_poses(poses_list): # Used for graphing purposes
    global poses_x
    global poses_y
    poses_x = [[i[0] for i in poses_list[p]] for p in map(str,[12,14,16,18,24,26,28])]
    for i in range(len(poses_x)):
        poses_x[i] = normalize_t(poses_x[i])
    poses_y = [[i[1] for i in poses_list[p]] for p in map(str,[12,14,16,18,24,26,28])]
    for i in range(len(poses_y)):
        poses_y[i] = normalize_t(poses_y[i])
    return poses_x, poses_y # list of numpy arrays

def stack_poses(poses_x,poses_y):
    output = dict()
    idx = 0
    for k in ['12','14','16','18','24','26','28']:
        output[k] = []
        for i in range(len(poses_x)):
            output[k].append((poses_x[idx][i], poses_y[idx][i]))
        idx += 1
    return output


def compare_ref(pose_list, ref_list):
    # For inputs, both are normalized to len(ref_list['24']) points
    
    # angle diffs during % of stroke + max diffs

    p_hip = pose_list['24']
    p_knee = pose_list['26']
    p_ankle = pose_list['28']
    p_shoulder = pose_list['12']
    r_hip = ref_list['24']
    r_knee = ref_list['26']
    r_ankle = ref_list['28']
    r_shoulder = ref_list['12']
    abs_knee_dist = []
    abs_body_dist = []
    for i in range(len(ref_list['24'])):
        p_knee_angle, p_body_angle = calc_angles(p_hip[i],p_knee[i],p_ankle[i],p_shoulder[i])
        r_knee_angle, r_body_angle = calc_angles(r_hip[i],r_knee[i],r_ankle[i],r_shoulder[i])
        abs_knee_dist.append(abs(p_knee_angle - r_knee_angle))
        abs_body_dist.append(abs(p_body_angle - r_body_angle))
    max_knee_diff = max(abs_knee_dist)
    max_knee_loc = abs_knee_dist.index(max_knee_diff)
    max_body_diff = max(abs_body_dist)
    max_body_loc = abs_body_dist.index(max_body_diff)
    #TODO: Add rowing phase segmentation
    print(f"Peak knee angle deviation of {round(max_knee_diff)} degrees at {max_knee_loc}% of the stroke")
    print(f"Peak body angle deviation of {round(max_body_diff)} degrees at {max_body_loc}% of the stroke")
    # overall stroke variation
    knee_rmse = math.sqrt(sum(((abs_knee_dist[j])**2) for j in range(len(ref_list['24']))) / len(ref_list['24']))
    body_rmse = math.sqrt(sum(((abs_knee_dist[j])**2) for j in range(len(ref_list['24']))) / len(ref_list['24']))
    print(f"Knee angle deviates {round(knee_rmse)}% from the reference stroke")
    print(f"Body angle deviates {round(body_rmse)}% from the reference stroke")

p_x, p_y = process_poses(list1)
list1 = stack_poses(p_x,p_y)
p_x, p_y = process_poses(list2)
list2 = stack_poses(p_x,p_y)
compare_ref(list1, list2)