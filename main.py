import mediapipe as mp
import cv2
import time
import keyboard

'''
TODO:
- Improve glitch filtering with Exponential Moving Average
- Determine how to extract youtube videos

'''



cap = cv2.VideoCapture('automation_test/2000m Row in 7 Minutes Row Along  Real Time Tips.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose(min_detection_confidence=0.85,min_tracking_confidence=0.85)
right_pose_list = [12,14,16,18,24,26,28]
right_connections = [(12,14),(14,16),(16,18),(12,24),(24,26),(26,28)]
left_pose_list = [11,13,15,17,23,25,27]
left_connections = [(11,13),(13,15),(15,17),(11,23),(23,25),(25,27)]
line_positions_dict = dict()
pos_locations_dict = dict()
left_visibility = None
right_visibility = None
finish = False
finish_frame = None
catch = False
catch_frame = None
shoulder = None
waited = 0
stroke_count = 0
frame_idx = 0
while True:
    success, img = cap.read()
    if not success:
        break
    # img = cv2.flip(img,-1)
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
       # mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx,cy = int(lm.x*w),int(lm.y*h)
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
        for i,f in curr_connections:
            i_cx,i_cy = line_positions_dict[i]
            f_cx,f_cy = line_positions_dict[f]
            cv2.line(img,(i_cx,i_cy),(f_cx,f_cy),(0,255,0),5)

    if catch and finish:
        stroke_rate = round(60*fps/(catch_frame-finish_frame),3)
        stroke_count += 1
        catch = False
        finish = False
        waited = 0
    if stroke_count:
        cv2.putText(img, f"Stroke rate: {stroke_rate}", (1200, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.putText(img, f"Stroke count: {stroke_count}", (1200, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    frame_idx += 1

