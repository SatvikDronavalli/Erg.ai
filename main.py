import mediapipe as mp
import cv2
import time
import keyboard

'''
Things to add:
    - Side switcher (detects which side is facing the camera and changes the landmarks used)
    


'''

cap = cv2.VideoCapture('Videos/IMG_8946 (1).MOV')

mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose(min_detection_confidence=0.85,min_tracking_confidence=0.85)
right_pose_list = [12,14,16,18,24,26,28]
right_connections = [(12,14),(14,16),(16,18),(12,24),(24,26),(26,28)]
left_pose_list = [11,13,15,17,23,25,27]
left_connections = [(11,13),(13,15),(15,17),(11,23),(23,25),(25,27)]
line_positions_dict = dict()
prev = time.time()
left_visibility = None
right_visibility = None
while True:
    if keyboard.is_pressed(" "):
        while True: # Add threading later
            time.sleep(0.01)
            if keyboard.is_pressed(" "):
                break
    success, img = cap.read()
    if len(img)==0:
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
                    print("right is greater")
                    curr_pose_list = right_pose_list
                    curr_connections = right_connections
                    break
                else:
                    print("left is greater")
                    curr_pose_list = left_pose_list
                    curr_connections = left_connections
                    break
        curr_pose_list = right_pose_list
        curr_connections = right_connections
        mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx,cy = int(lm.x*w),int(lm.y*h)
            line_positions_dict.update({id: (cx,cy)})
            if id in curr_pose_list:
                # print(id, lm.visibility)
                cv2.circle(img, (cx,cy), 5, (255,0,0),cv2.FILLED)
        for i,f in left_connections:
            i_cx,i_cy = line_positions_dict[i]
            f_cx,f_cy = line_positions_dict[f]
            cv2.line(img,(i_cx,i_cy),(f_cx,f_cy),(0,255,0),5)
    curr = time.time()
    fps = 1/(curr-prev)
    prev = curr
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

