import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

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
        if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate((pose, face, lh, rh))

def create_folders(path, actions, num_videos, num_frames):
    """Creates folders if they don't exist"""
    for action in actions:
        for sequence in range(num_videos):
            try:
                os.makedirs(os.path.join(path, action, str(sequence)))
            except:
                pass

def train_data(data_path, actions: np.array(str), num_videos, num_frames):
    """Self made training data for each gesture"""
    create_folders(data_path, actions, num_videos, num_frames)
    cap = cv2.VideoCapture(0)
    # set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as model:
        # can change loop condition later just opens frame
        for action in actions:
            for sequence in range(num_videos):
                for frame_num in range(num_frames):
                    # read 1 frame from camera
                    ret, frame = cap.read()
                    # incorrect frame read
                    if not ret:
                        print("Unable to capture frame.")
                        break

                    # make detections and draw them on tracking_frame
                    image, results = mediapipe_detection(frame, model)
                    draw_landmarks(image, results)
                    #TRAINING GESTURES YOURSELF
                    if frame_num == 0:
                        cv2.putText(image, 'Starting Collection', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow("Tracking Feed", image)
                        cv2.imshow("Regular Feed", frame)
                        # take a break for 2 seconds every start of video
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow("Tracking Feed", image)
                        cv2.imshow("Regular Feed", frame)

                    # extracts keypoints and saves them to path
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    # display

                    # press q while in window to exit loop
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
        # remove video cap object and close the window
        cap.release()
        cv2.destroyAllWindows()
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame



if __name__ == "__main__":
    # num videos
    num_videos = 30
    # num frames
    num_frames = 30
    actions = np.array(['hello', 'iloveyou', 'promise', 'thanks'])


    curr_model = load_model('action.h5')
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.7
    res = np.zeros(len(actions))
    cap = cv2.VideoCapture(0)
    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (111, 111, 111)]

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            ret, frame = cap.read()

            image, results = mediapipe_detection(frame, holistic)

            draw_landmarks(image, results)


            #predictions
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = curr_model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
            uniques = np.unique(predictions[-10:])
            if len(uniques) > 0 and uniques[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:

                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Tracking Feed", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
