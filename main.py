import mediapipe as mp
import cv2
import math

cap = cv2.VideoCapture('videos/satvik_erg.MOV')
fps = cap.get(cv2.CAP_PROP_FPS)
mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose(min_detection_confidence=0.85,min_tracking_confidence=0.85)
right_pose_list = [12,14,16,18,24,26,28]
right_connections = [(12,14),(14,16),(16,18),(12,24),(24,26),(26,28)]
left_pose_list = [11,13,15,17,23,25,27]
left_connections = [(11,13),(13,15),(15,17),(11,23),(23,25),(25,27)]
prev_positions_ema = dict()
line_positions_dict = dict()
pos_locations_dict = dict()
left_visibility = None
right_visibility = None
finish = False
finish_frame = None
catch = False
catch_frame = None
shoulder = None
curr_pose_list = None
curr_connections = None
first_frame = None
body_direction = 0
initial_height = 0
waited = 0
stroke_count = 0
frame_idx = 0

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

def make_vert_dashed_line(dashes,GAP_MULTIPLIER,start,end,img,color,thickness):
    dist_y = (end[1]-start[1])/(dashes+(dashes-1)*GAP_MULTIPLIER)
    # cv2.line(img, hip, ghost_body_pos, (0, 255, 0), 5)
    for i in range(1, dashes+1):
        start_seg_y = round(start[1]+((i-1)*(dist_y+dist_y*GAP_MULTIPLIER)))
        end_seg_y = round(start_seg_y+dist_y)
        cv2.line(img,(start[0],start_seg_y),(start[0],end_seg_y),color,thickness)

def exponential_moving_average(cx,cy,px,py,a=0.3):
    # From testing, a=0.3 minimized knee + body angle variance
    return int(cx*a+px*(1-a)),int(cy*a+py*(1-a))

alphas = [0.2+i*0.025 for i in range(0,11)] # 10 strokes in test video
glitches = dict()
max_knee_glitch = -1
max_body_glitch = -1
prev_knee_angle = 0
knee_angle = 0
prev_body_angle = 0
body_angle = 0
a_idx = 0
ended_stroke = False

while True:
    curr_alpha = 0.275 # alphas[a_idx]
    ended_stroke = False
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img,-1)
    img = cv2.resize(img, (1700,1000))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        if not left_visibility and not right_visibility:
            for id, lm in enumerate(results.pose_landmarks.landmark):
                if id == 25:
                    left_visibility = lm.visibility
                elif id == 26:
                    right_visibility = lm.visibility
                if (left_visibility and right_visibility) and right_visibility > left_visibility:
                    curr_pose_list = right_pose_list
                    curr_connections = right_connections
                    shoulder = 12
                    break
                elif left_visibility and right_visibility:
                    curr_pose_list = left_pose_list
                    curr_connections = left_connections
                    shoulder = 11
                    break
        if not curr_pose_list:
            break
       # mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx,cy = int(lm.x*w),int(lm.y*h)
            if id in prev_positions_ema:
                px,py = prev_positions_ema[id]
                cx,cy = exponential_moving_average(cx,cy,px,py,curr_alpha)
            line_positions_dict.update({id: (cx,cy)})
            if id in curr_pose_list:
                # print(id, lm.visibility)
                cv2.circle(img, (cx,cy), 10, (255,0,0),cv2.FILLED)
                if not finish:
                    if id not in pos_locations_dict.keys():
                        pos_locations_dict.update({id: [(cx,cy)]})
                    else:
                        pos_locations_dict[id].append((cx,cy))
                        if id == 12 and len(pos_locations_dict[shoulder]) >= 7:
                            # adjust this based on glitches
                            if pos_locations_dict[shoulder][-1][0] > pos_locations_dict[shoulder][-7][0]:
                                finish_frame = frame_idx
                                finish = True
                        elif id == 11 and len(pos_locations_dict[shoulder]) >= 7:
                            if pos_locations_dict[shoulder][-1][0] < pos_locations_dict[shoulder][-7][0]:
                                finish_frame = frame_idx
                                finish = True
                elif finish and not catch:
                    pos_locations_dict[id].append((cx,cy))
                    if waited < 10 and id == shoulder:
                        waited += 1
                    else:
                        if id == 12:
                            # adjust this based on glitches
                            if pos_locations_dict[shoulder][-1][0] < pos_locations_dict[shoulder][-7][0]:
                                catch_frame = frame_idx
                                catch = True
                        elif id == 11:
                            if pos_locations_dict[shoulder][-1][0] > pos_locations_dict[shoulder][-7][0]:
                                catch_frame = frame_idx
                                catch = True
            prev_positions_ema[id] = (cx,cy)
        for i,f in curr_connections:
            i_cx,i_cy = line_positions_dict[i]
            f_cx,f_cy = line_positions_dict[f]
            cv2.line(img,(i_cx,i_cy),(f_cx,f_cy),(0,255,0),5)
    else:
        continue
    if catch and finish:
        stroke_rate = round(60*fps/(catch_frame-finish_frame),3)
        stroke_count += 1
        ended_stroke = True
        catch = False
        finish = False
        waited = 0
    if stroke_count:
        cv2.putText(img, f"Stroke rate: {stroke_rate}", (1200, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.putText(img, f"Stroke count: {stroke_count}", (1200, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    # cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    # Add circle sector graphing here
    h,w,c = img.shape
    VERT_SCALE = 1.2
    if curr_pose_list == right_pose_list:
        hip = (line_positions_dict[24][0],line_positions_dict[24][1])
        knee = (line_positions_dict[26][0],line_positions_dict[26][1])
        ankle = (line_positions_dict[28][0],line_positions_dict[28][1])
        shoulder_pos = (line_positions_dict[12][0],line_positions_dict[12][1])
        if not first_frame:
            first_frame = True
            initial_height = round(line_positions_dict[24][1] + VERT_SCALE*(line_positions_dict[12][1]-line_positions_dict[24][1]))
        ghost_body_pos = (line_positions_dict[24][0],initial_height)
        a,b,c = get_knee_lengths(hip=hip,knee=knee,ankle=ankle)
        d,e = get_body_lengths(hip=hip,shoulder=shoulder_pos)
        knee_angle = round(find_angle(a,b,c)*(180/math.pi),1)
        if shoulder_pos[0] > hip[0]:
            body_direction = 1
        elif shoulder_pos[0] == hip[0]:
            body_direction = 0
        else:
            body_direction = -1
        body_angle = round((180-round(math.acos(e / d) * (180 / math.pi), 3)),1)*body_direction
        # knee sector graphing
        SECTOR_RADIUS_DIVISOR = 3
        len_knee_to_hip = abs(math.dist(knee,hip))
        len_knee_to_ankle = abs(math.dist(knee,ankle))
        len_hip_to_shoulder = abs(math.dist(hip,shoulder_pos))
        len_hip_to_ghost = abs(math.dist(hip,ghost_body_pos))
        radius_k = min(len_knee_to_hip,len_knee_to_ankle)/SECTOR_RADIUS_DIVISOR
        radius_h = min(len_hip_to_shoulder,len_hip_to_ghost)/SECTOR_RADIUS_DIVISOR
        if ankle[0] < knee[0]:
            x_axis_length = math.dist(knee, (knee[0] - (knee[0] - ankle[0]), knee[1]))
            initial_angle = 180 - (math.acos(x_axis_length / len_knee_to_ankle) * (180 / math.pi))
        else:
            x_axis_length = math.dist(knee,(knee[0]+(ankle[0]-knee[0]),knee[1]))
            initial_angle = math.acos(x_axis_length/len_knee_to_ankle)*(180/math.pi)
        INITIAL_ANGLE_OFFSET = 1.25
        ARC_ANGLE_OFFSET = 2
        overlay = img.copy()
        cv2.ellipse(img,knee,(round(radius_k),round(radius_k)),initial_angle+INITIAL_ANGLE_OFFSET,0,knee_angle-ARC_ANGLE_OFFSET,(160,170,80),-1)
        cv2.ellipse(img, hip, (round(radius_h), round(radius_h)), -90 + INITIAL_ANGLE_OFFSET, 0,
                    body_angle - ARC_ANGLE_OFFSET, (160, 170, 80), -1)
        make_vert_dashed_line(4,2,hip,ghost_body_pos,img,(0,255,0),5)
        ANGLE_INDICATOR_DISTANCE = 1.5
        TEXT_OFFSET = 300
        (size,baseline) = cv2.getTextSize(f"Knee angle: {knee_angle}",cv2.FONT_HERSHEY_PLAIN,2,3)
        text_w_k,text_h_k = size
        (size_b, baseline_b) = cv2.getTextSize(f"Body angle: {body_angle}", cv2.FONT_HERSHEY_PLAIN, 2, 3)
        text_w_b, text_h_b = size_b
        text_pos_x_k = max(0,knee[0] + round(ANGLE_INDICATOR_DISTANCE*radius_k*math.cos((initial_angle+(knee_angle/2))*(math.pi/180))) - int(text_w_k/2))
        text_pos_y_k = max(0, knee[1] + round(ANGLE_INDICATOR_DISTANCE*radius_k*math.sin((initial_angle+(knee_angle/2))*(math.pi/180))))
        text_pos_x_h = max(0, hip[0] + round(
            ANGLE_INDICATOR_DISTANCE * radius_h * math.cos(((body_angle / 2) - 90) * (math.pi / 180))) - int(
            text_w_b / 2))
        text_pos_y_h = max(0, hip[1] + round(
            ANGLE_INDICATOR_DISTANCE * radius_h * math.sin(((body_angle / 2) - 90) * (math.pi / 180))))
        # body sector graphing
        '''
        len_hip_to_shoulder = abs(math.dist(hip,shoulder_pos))
        radius_h = min(len_hip_to_shoulder,len_knee_to_hip)
        if knee[1] > ankle[1]:
            x_axis_length = math.dist(knee,) '''
        TRANSPARENCY_ALPHA = 0.5
        translucent_arc = cv2.addWeighted(overlay,TRANSPARENCY_ALPHA,img,1-TRANSPARENCY_ALPHA,0,img)
        cv2.putText(img, f"Knee angle: {knee_angle}", (text_pos_x_k, text_pos_y_k), cv2.FONT_HERSHEY_PLAIN, 2, (245, 245, 245),
                    3)
        cv2.putText(img, f"Knee angle: {knee_angle}", (text_pos_x_k+2, text_pos_y_k+2), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 0, 0),
                    1)
        cv2.putText(img, f"Body angle: {body_angle}", (text_pos_x_h, text_pos_y_h), cv2.FONT_HERSHEY_PLAIN, 2,
                    (245, 245, 245),
                    3)
        cv2.putText(img, f"Body angle: {body_angle}", (text_pos_x_h + 2, text_pos_y_h + 2), cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 0, 0),
                    1)
    elif curr_pose_list == left_pose_list: #TODO: patch this side
        hip = (line_positions_dict[23][0], line_positions_dict[23][1])
        knee = (line_positions_dict[25][0], line_positions_dict[25][1])
        ankle = (line_positions_dict[27][0], line_positions_dict[27][1])
        shoulder_pos = (line_positions_dict[11][0], line_positions_dict[11][1])
        a,b,c = get_knee_lengths(hip=hip,knee=knee,ankle=ankle)
        d,e = get_body_lengths(hip=hip,shoulder=shoulder_pos,knee=knee)
        knee_angle = round(find_angle(a,b,c)*(180/math.pi),3)
        body_angle = round(find_angle(d,e,f)*(180/math.pi),3)
    max_knee_glitch = max(max_knee_glitch, knee_angle - prev_knee_angle)
    max_body_glitch = max(max_body_glitch, body_angle - prev_body_angle)
    if ended_stroke and stroke_count > 1: # Cycle through alphas here
        # a_idx += 1
        glitches[curr_alpha] = (round(max_knee_glitch,2),round(max_body_glitch,2))
        max_knee_glitch = 0
        max_body_glitch = 0
        # print(f"(knee,body) glitch at a={round(curr_alpha,3)}: {glitches[curr_alpha]}")
    # cv2.putText(img, f"Knee angle: {knee_angle}", (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    # cv2.putText(img, f"Body angle: {body_angle}", (70, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    if a_idx > 0:
        cv2.putText(img, f"Stroke variance: {glitches[alphas[a_idx - 1]]}", (70, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.putText(img, f"Stroke alpha: {round(alphas[a_idx - 1],3)}", (70, 200), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    frame_idx += 1
    prev_knee_angle = knee_angle
    prev_body_angle = body_angle

