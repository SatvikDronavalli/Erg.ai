import mediapipe as mp
import cv2

'''
TODO:
- Improve glitch filtering with Exponential Moving Average
- Determine how to extract youtube videos

'''


def check_validity(path):

    # Environment setup

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    mpPose = mp.solutions.pose
    pose = mpPose.Pose(min_detection_confidence=0.85,min_tracking_confidence=0.85)
    right_pose_list = [12,14,16,18,24,26,28]
    left_pose_list = [11,13,15,17,23,25,27]
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
    prev_stroke_rate = None
    temp = []
    # Loop for video processing

    while True:
        if stroke_count >= 10:  # Heuristic, 10 contiguous strokes
            ten_good_strokes = temp
            return True, ten_good_strokes
        success, img = cap.read()
        if not success or img is None:
            break # End of video

        img = cv2.flip(img,-1)
        scale_width = 640
        h,w,_ = img.shape
        scale_height = int(h * scale_width / w)
        img = cv2.resize(img, (scale_width,scale_height))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks: # If landmarks are detected

            # Side selection based on visibility

            if left_visibility is None and right_visibility is None:
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    if id == 25:
                        left_visibility = lm.visibility
                    elif id == 26:
                        right_visibility = lm.visibility
                    if left_visibility is not None and right_visibility is not None:
                        if right_visibility > left_visibility:
                            curr_pose_list = right_pose_list
                            shoulder = 12
                            break
                        elif right_visibility <= left_visibility:
                            curr_pose_list = left_pose_list
                            shoulder = 11
                            break
            visibilities = [results.pose_landmarks.landmark[lm].visibility for lm in curr_pose_list]
            if sum(visibilities) / len(curr_pose_list) < 0.85:
                print("reset")
                stroke_count = 0
                pos_locations_dict = {}
                finish = False
                catch = False
                waited = 0
                continue

            # Determine stroke duration by logging catch and finish times

            for id, lm in enumerate(results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                if id in curr_pose_list: # if the position is relevant

                    # Logs time that the finish is reached

                    if not finish:
                        if id not in pos_locations_dict.keys():
                            pos_locations_dict.update({id: [(cx,cy)]})
                        else:
                            pos_locations_dict[id].append((cx,cy))
                            # Right side
                            if id == 12 and len(pos_locations_dict[shoulder]) >= 7:
                                # adjust this based on glitches
                                if pos_locations_dict[shoulder][-1][0] > pos_locations_dict[shoulder][-7][0]:
                                    finish_frame = frame_idx
                                    finish = True
                            # Left side
                            elif id == 11 and len(pos_locations_dict[shoulder]) >= 7:
                                if pos_locations_dict[shoulder][-1][0] < pos_locations_dict[shoulder][-7][0]:
                                    finish_frame = frame_idx
                                    finish = True

                    # Logs time that the catch is reached (mirrored version of finish logic)

                    elif finish and not catch:
                        pos_locations_dict[id].append((cx,cy))
                        if waited < 10 and id == shoulder: # Accounts for directional change at the finish
                            waited += 1
                        else:
                            # Right side
                            if id == 12:
                                # adjust this based on glitches
                                if pos_locations_dict[shoulder][-1][0] < pos_locations_dict[shoulder][-7][0]:
                                    catch_frame = frame_idx
                                    catch = True
                            # Left side
                            elif id == 11:
                                if pos_locations_dict[shoulder][-1][0] > pos_locations_dict[shoulder][-7][0]:
                                    catch_frame = frame_idx
                                    catch = True
        if catch and finish:
            stroke_rate = round(60*fps/(catch_frame-finish_frame),3)
            if prev_stroke_rate is not None and abs(prev_stroke_rate-stroke_rate) <= 7:
                stroke_count += 1
                temp.append(pos_locations_dict)
                print(f"{stroke_count} contiguous {'stroke' if stroke_count == 1 else 'strokes'}")
            pos_locations_dict = {}
            catch = False
            finish = False
            waited = 0
            prev_stroke_rate = stroke_rate
        cv2.waitKey(1)
        frame_idx += 1
    return False, []

print(check_validity("2000m Row in 7 Minutes Row Along  Real Time Tips.mp4"))