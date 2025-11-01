import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic #model
mp_drawing = mp.solutions.drawing_utils #drawing utilities
mp_face_mesh = mp.solutions.face_mesh
DATA_PATH = os.path.join('MP_Data')


def mediapipe_detection(image, model):
    """detects parts of body in image"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # color conversion from bgr to rgb
    image.flags.writeable = False   #image isnt writeable
    results = model.process(image) #prediction
    image.flags.writeable = True #writeable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # converts it back
    return image, results

def draw_landmarks(image, results):
    """Draws the connections between body parts on image"""
    mp_drawing.draw_landmarks(image, landmark_list=results.face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION,
                              landmark_drawing_spec= mp_drawing.DrawingSpec(color= (80, 110, 10), thickness=1, circle_radius=1),
                              connection_drawing_spec= mp_drawing.DrawingSpec(color= (80, 255, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              landmark_drawing_spec= mp_drawing.DrawingSpec(color= (158, 37, 0), thickness=2, circle_radius=1),
                              connection_drawing_spec= mp_drawing.DrawingSpec(color= (255, 121, 80), thickness=2, circle_radius=1)
                              )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec= mp_drawing.DrawingSpec(color= (18, 43, 222), thickness=2, circle_radius=2),
                              connection_drawing_spec= mp_drawing.DrawingSpec(color= (18, 43, 222), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(18, 43, 222), thickness=2,circle_radius=2),
                              connection_drawing_spec=mp_drawing.DrawingSpec(color=(18, 43, 222), thickness=2, circle_radius=2))

def extract_keypoints(results):
    """
    extracts keypoints from frame passed in
    return format:
    nparray (pose, face, lh, rh)
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(448*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate((pose, face, lh, rh))


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    #set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as model:
        #can change loop condition later just opens frame
        while cap.isOpened():
            #read 1 frame from camera
            ret, frame = cap.read()
            #incorrect frame read
            if not ret:
                print("Unable to capture frame.")
                break

            #make detections and draw them on tracking_frame
            image, results = mediapipe_detection(frame, model)
            draw_landmarks(image, results)
            #display frame
            cv2.imshow("Tracking Feed", image)
            cv2.imshow("Regular Feed", frame)
            #press q while in window to exit loop
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        #remove video cap object and close the window
        cap.release()
        cv2.destroyAllWindows()

    #actions that we are gonna try and detect
    actions = np.array(['hello', 'thanks', 'iloveyou'])
    #num videos
    num_videos = 30
    #num frames
    num_frames = 30

    for action in actions:
        for sequence in range(num_videos):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass
