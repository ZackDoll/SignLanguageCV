import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def mediapipe_detection(image, model):
    """Detects parts of body in image"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    """Draws the connections between body parts on image"""
    mp_drawing.draw_landmarks(
        image, landmark_list=results.face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(80, 110, 10), thickness=1, circle_radius=1),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(80, 255, 121), thickness=1, circle_radius=1))

    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(158, 37, 0), thickness=2, circle_radius=1),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 121, 80), thickness=2, circle_radius=1))

    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(18, 43, 222), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(18, 43, 222), thickness=2, circle_radius=2))

    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(18, 43, 222), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(18, 43, 222), thickness=2, circle_radius=2))


def extract_keypoints(results):
    """Extract keypoints from frame"""
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z]
                     for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate((pose, face, lh, rh))


def prob_viz(res, actions, input_frame, colors, top_k=5):
    """Visualize top predictions with probability bars"""
    output_frame = input_frame.copy()

    # Get top k predictions
    top_indices = np.argsort(res)[-top_k:][::-1]

    for i, idx in enumerate(top_indices):
        prob = res[idx]
        action = actions[str(idx)] if str(idx) in actions else f"Class {idx}"

        # Draw probability bar
        cv2.rectangle(output_frame, (0, 60 + i * 40),
                      (int(prob * 300), 90 + i * 40),
                      colors[i % len(colors)], -1)

        # text writer
        cv2.putText(output_frame, f"{action}: {prob:.2f}",
                    (0, 85 + i * 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame


def run_msasl_inference(model_path='msasl_model.h5',
                        classes_path='msasl_classes.json',
                        threshold=0.7, sequence_length=30):
    """Run real-time sign language detection"""

    # load model and classes
    print("Loading model...")
    model = load_model(model_path)
    print("Loading classes...")
    with open(classes_path, 'r') as f:
        actions_raw = json.load(f)

    # Determine the format and create index -> name mapping
    actions = {}
    sample_key = list(actions_raw.keys())[0]

    if sample_key.isdigit():  # ✅ CHECK if it's a digit string FIRST
        # Keys are string numbers: {"0": "HELLO"} -> {0: "HELLO"}
        actions = {int(k): v for k, v in actions_raw.items()}
        print("Converted string number keys to integers")
    else:
        # Keys are sign names: {"HELLO": 0} -> {0: "HELLO"} (reverse)
        actions = {v: k for k, v in actions_raw.items()}  # ✅ REVERSE the mapping
        print("Reversed mapping: sign names to indices")

    print(f"Model loaded with {len(actions)} classes")
    print(f"Sample mappings: {dict(list(actions.items())[:3])}")
    print(f"Model loaded with {len(actions)} classes")

    sequence = []
    sentence = []
    predictions = []
    threshold = threshold

    # Colors for top 5 probabilities
    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245),
              (111, 111, 111), (245, 16, 117)]

    #start video
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:

        print("Starting detection... Press 'q' to quit")

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            #make detections
            image, results = mediapipe_detection(frame, holistic)

            # draw landmarks
            draw_landmarks(image, results)

            # extract keypoints and add to sequence
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]

            # make prediction when we have enough frames
            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0),
                                    verbose=0)[0]
                predicted_class = np.argmax(res)
                confidence = res[predicted_class]

                # If keys are strings like "HELLO", converts to reverse mapping
                if isinstance(list(actions.keys())[0], str):
                    actions = {int(k): v for k, v in actions.items()}  # Try to convert
                    # OR creates reverse mapping if needed

                # Try int key first, then string key
                if predicted_class in actions:
                    action_name = actions[predicted_class]
                elif str(predicted_class) in actions:
                    action_name = actions[str(predicted_class)]

                print(f"Prediction: {action_name} "
                      f"(confidence: {confidence:.2f})")

                predictions.append(predicted_class)

                # check for consistent predictions
                if len(predictions) >= 10:
                    recent_predictions = predictions[-10:]
                    unique, counts = np.unique(recent_predictions,
                                               return_counts=True)

                    if counts.max() >= 7 and confidence > threshold:
                        # Get the most common prediction
                        most_common_class = unique[np.argmax(counts)]

                        # Get action name - handle both int and string keys
                        if most_common_class in actions:
                            predicted_action = actions[most_common_class]
                        elif str(most_common_class) in actions:
                            predicted_action = actions[str(most_common_class)]
                        else:
                            predicted_action = f"Class {most_common_class}"

                        # add to sentence if new
                        if len(sentence) == 0 or predicted_action != sentence[-1]:
                            sentence.append(predicted_action)

                #keep sentence length to 5 ( can change if needed)
                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # visualizer
                image = prob_viz(res, actions, image, colors)

            # sentence bar
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2, cv2.LINE_AA)

            # show final frame
            cv2.imshow("MS-ASL Detection", image)

            # q to quit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # run for trained model
    run_msasl_inference(
        model_path='asl_citizen_model.h5',
        classes_path='asl_citizen_glosses.json',
        threshold=0.7,
        sequence_length=30
    )