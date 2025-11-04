import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


DATA_PATH = os.path.join('MP_Data')
"""
def train_model(data_path, actions: np.array(str), num_videos, num_frames, epochs = 800):
    #trains model
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
"""
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report
from collections import Counter


class MSASLTrainer:
    """Train model on MS-ASL dataset"""

    def __init__(self, processed_dir='MP_Data_MSASL', dataset_dir='MSASL_data'):
        self.processed_dir = processed_dir
        self.dataset_dir = dataset_dir
        self.num_frames = 30
        self.num_features = 1662

    def load_classes(self):
        """Load class names from MS-ASL"""
        classes_path = os.path.join(self.dataset_dir, 'MSASL_classes.json')
        with open(classes_path, 'r') as f:
            classes = json.load(f)
        return classes

    def load_processed_data(self, split='train', max_samples=None,
                            class_subset=None):
        """Load preprocessed keypoint data"""
        split_dir = os.path.join(self.processed_dir, split)

        sequences = []
        labels = []

        # Get all label directories
        label_dirs = [d for d in os.listdir(split_dir)
                      if os.path.isdir(os.path.join(split_dir, d))]

        # Filter by class subset if provided
        if class_subset:
            label_dirs = [d for d in label_dirs if int(d) in class_subset]

        print(f"Loading {len(label_dirs)} classes from {split} split...")

        for label_dir in sorted(label_dirs, key=lambda x: int(x)):
            label = int(label_dir)
            label_path = os.path.join(split_dir, label_dir)

            # Load all .npy files in this label directory
            npy_files = [f for f in os.listdir(label_path) if f.endswith('.npy')]

            for npy_file in npy_files:
                if max_samples and len(sequences) >= max_samples:
                    break

                npy_path = os.path.join(label_path, npy_file)
                try:
                    keypoints = np.load(npy_path)
                    if keypoints.shape == (self.num_frames, self.num_features):
                        sequences.append(keypoints)
                        labels.append(label)
                except Exception as e:
                    print(f"Error loading {npy_path}: {e}")
                    continue

            if max_samples and len(sequences) >= max_samples:
                break

        print(f"Loaded {len(sequences)} sequences")
        print(f"Label distribution: {Counter(labels)}")

        return np.array(sequences), np.array(labels)

    def build_model(self, num_classes, lstm_units=[64, 128, 64],
                    dense_units=[64, 32], dropout_rate=0.3):
        """Build LSTM model architecture"""
        model = Sequential()

        #LSTM layers
        model.add(LSTM(lstm_units[0], return_sequences=True,
                       activation='relu',
                       input_shape=(self.num_frames, self.num_features)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        for units in lstm_units[1:-1]:
            model.add(LSTM(units, return_sequences=True, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        model.add(LSTM(lstm_units[-1], return_sequences=False, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        #Dense layers
        for units in dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        # Output layer
        model.add(Dense(num_classes, activation='softmax'))

        return model

    def train_model(self, X_train, y_train, X_val=None, y_val=None,
                    epochs=100, batch_size=32, model_save_path='msasl_model.h5'):
        """Train the model"""

        num_classes = y_train.shape[1]
        print(f"Training model with {num_classes} classes...")

        model = self.build_model(num_classes)

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

        #idk what summary is but its in the documentation so ill take it
        model.summary()

        # make logs
        log_dir = os.path.join('logs', 'msasl_logs')
        os.makedirs(log_dir, exist_ok=True)

        callbacks = [
            TensorBoard(log_dir=log_dir),
            ModelCheckpoint(model_save_path, save_best_only=True,
                            monitor='val_categorical_accuracy' if X_val is not None
                            else 'categorical_accuracy'),
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss',
                          patience=15, restore_best_weights=True)
        ]

        # validations
        validation_data = (X_val, y_val) if X_val is not None else None

        # train step
        history = model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        return model, history

    def evaluate_model(self, model, X_test, y_test, classes):
        """Evaluate model performance"""
        #predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        #aim for 75 - 85% 100 is overfitting 65 is under
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        print(f"\nTest Accuracy: {accuracy:.4f}")

        conf_matrix = multilabel_confusion_matrix(y_true_classes, y_pred_classes)
        print("\nConfusion Matrix:")
        print(conf_matrix)

        # Classification report
        unique_labels = sorted(list(set(y_true_classes)))
        target_names = [classes[i] for i in unique_labels]
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes,
                                    labels=unique_labels,
                                    target_names=target_names))

        return accuracy, y_pred_classes

    def prepare_training_data(self, class_subset=None, max_samples_per_split=None):
        """Load and prepare all training data"""
        classes = self.load_classes()

        # Load data
        X_train, y_train = self.load_processed_data('train', max_samples_per_split,
                                                    class_subset)
        X_val, y_val = self.load_processed_data('val', max_samples_per_split,
                                                class_subset)
        X_test, y_test = self.load_processed_data('test', max_samples_per_split,
                                                  class_subset)

        # map labels to proper videos in subset
        if class_subset:
            unique_labels = sorted(list(set(y_train)))
            label_map = {old_label: new_label
                         for new_label, old_label in enumerate(unique_labels)}

            y_train = np.array([label_map[label] for label in y_train])
            y_val = np.array([label_map[label] for label in y_val])
            y_test = np.array([label_map[label] for label in y_test])

            # update class list
            classes = [classes[i] for i in unique_labels]

        # convert everything to categorical data
        num_classes = len(set(y_train))
        y_train = to_categorical(y_train, num_classes)
        y_val = to_categorical(y_val, num_classes)
        y_test = to_categorical(y_test, num_classes)

        print(f"\nDataset shapes:")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
        print(f"Number of classes: {num_classes}")

        return X_train, y_train, X_val, y_val, X_test, y_test, classes


def train_msasl_model(class_subset=None, max_samples=None, epochs=100):
    """Main training function"""

    trainer = MSASLTrainer()

    # data prep
    X_train, y_train, X_val, y_val, X_test, y_test, classes = \
        trainer.prepare_training_data(class_subset, max_samples)

    # train using keras
    model, history = trainer.train_model(
        X_train, y_train, X_val, y_val,
        epochs=epochs,
        batch_size=32,
        model_save_path='msasl_model.h5'
    )

    # eval
    accuracy, predictions = trainer.evaluate_model(model, X_test, y_test, classes)
    print("Accuracy: ", accuracy)

    # save the mapping
    class_mapping = {i: classes[i] for i in range(len(classes))}
    with open('msasl_classes.json', 'w') as f:
        json.dump(class_mapping, f)

    print("\nModel saved to msasl_model.h5")
    print("Classes saved to msasl_classes.json")

    return model, classes


if __name__ == "__main__":

    # train full dataset
    print("Training on full MS-ASL dataset...")
    model, classes = train_msasl_model(epochs=100) #mess with epochs to get good value
"""
if __name__ == "__main__":
    # num videos
    num_videos = 30
    # num frames
    num_frames = 30
    actions = np.array(['hello', 'iloveyou', 'promise', 'thanks', 'no_sign'])

    #train_data(DATA_PATH, actions, num_videos, num_frames)

    curr_model, test_x, test_y = train_model(DATA_PATH, actions, num_videos, num_frames)
    res = curr_model.predict(test_x)
    print(actions[np.argmax(res[0])])
    print(actions[np.argmax(test_y[0])])

    curr_model.save('action.h5')
    #eval if needed
    
    yhat = curr_model.predict(test_x)
    ytrue = np.argmax(test_y, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    print(multilabel_confusion_matrix(ytrue, yhat))
    print(accuracy_score(ytrue, yhat))
    del curr_model
"""