import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


DATA_PATH = os.path.join('MP_Data')

def train_model(data_path, actions: np.array(str), num_videos, num_frames, epochs = 250):
    """Trains the model"""
    label_map = {label:num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for sequence in range(num_videos):
            window = []
            for frame_num in range(num_frames):
                #file, column, folder, number
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    X = np.array(sequences)
    Y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05)

    log_dir = os.path.join('logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    #neural network
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation = 'relu', input_shape = (30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    #make sure to not return sequences when switching from LSTM to Dense
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    #to check on TensorBoard state go to Logs file in project folder and run tensorboard --logdir=. in terminal
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=epochs, callbacks=[tb_callback])

    return model, X_test, y_test



if __name__ == "__main__":
    # num videos
    num_videos = 30
    # num frames
    num_frames = 30
    actions = np.array(['hello', 'iloveyou', 'promise', 'thanks'])

    #train_data(DATA_PATH, actions, num_videos, num_frames)

    curr_model, test_x, test_y = train_model(DATA_PATH, actions, num_videos, num_frames)
    res = curr_model.predict(test_x)
    print(actions[np.argmax(res[0])])
    print(actions[np.argmax(test_y[0])])

    curr_model.save('action.h5')
    #eval if needed
    """
    """
    yhat = curr_model.predict(test_x)
    ytrue = np.argmax(test_y, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    print(multilabel_confusion_matrix(ytrue, yhat))
    print(accuracy_score(ytrue, yhat))
    del curr_model
