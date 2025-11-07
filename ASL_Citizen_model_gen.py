import numpy as np
import os
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from collections import Counter


class ASLCitizenTrainer:
    """Trainer with maximum regularization and optimization"""

    def __init__(self, processed_dir='Processed_Keypoints', dataset_dir='ASL_Citizen'):
        self.processed_dir = processed_dir
        self.dataset_dir = dataset_dir
        self.num_frames = 30
        self.num_features = 1662

    def load_processed_data(self, split='train', max_samples=None, gloss_subset=None):
        """Load preprocessed keypoint data"""
        split_dir = os.path.join(self.processed_dir, split)

        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Processed directory not found: {split_dir}")

        sequences = []
        labels = []

        gloss_dirs = [d for d in os.listdir(split_dir)
                      if os.path.isdir(os.path.join(split_dir, d))]

        if gloss_subset:
            gloss_dirs = [d for d in gloss_dirs if d in gloss_subset]

        print(f"Loading {len(gloss_dirs)} signs from {split} split...")

        for gloss_dir in sorted(gloss_dirs):
            gloss_path = os.path.join(split_dir, gloss_dir)
            npy_files = [f for f in os.listdir(gloss_path) if f.endswith('.npy')]

            for npy_file in npy_files:
                if max_samples and len(sequences) >= max_samples:
                    break

                npy_path = os.path.join(gloss_path, npy_file)
                try:
                    keypoints = np.load(npy_path)
                    if keypoints.shape == (self.num_frames, self.num_features):
                        sequences.append(keypoints)
                        labels.append(gloss_dir)
                except Exception as e:
                    print(f"Error loading {npy_path}: {e}")
                    continue

            if max_samples and len(sequences) >= max_samples:
                break

        print(f"Loaded {len(sequences)} sequences")
        return np.array(sequences), np.array(labels)

    def build_model(self, num_classes, model_size='small'):
        """Build heavily regularized model"""

        if model_size == 'small':
            lstm_units = [32, 64, 32]
            dense_units = [32]
            dropout_rate = 0.6  # VERY high dropout
            l2_reg = 0.02  # Strong L2
        elif model_size == 'tiny':
            lstm_units = [16, 32, 16]
            dense_units = [16]
            dropout_rate = 0.65
            l2_reg = 0.02
        else:
            lstm_units = [64, 128, 64]
            dense_units = [64, 32]
            dropout_rate = 0.5
            l2_reg = 0.015

        model = Sequential()

        # LSTM layers with heavy regularization
        model.add(LSTM(lstm_units[0], return_sequences=True,
                       activation='relu',
                       kernel_regularizer=l2(l2_reg),
                       recurrent_regularizer=l2(l2_reg),
                       dropout=0.2,  # Recurrent dropout
                       recurrent_dropout=0.2,
                       input_shape=(self.num_frames, self.num_features)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        for units in lstm_units[1:-1]:
            model.add(LSTM(units, return_sequences=True, activation='relu',
                           kernel_regularizer=l2(l2_reg),
                           recurrent_regularizer=l2(l2_reg),
                           dropout=0.2,
                           recurrent_dropout=0.2))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        model.add(LSTM(lstm_units[-1], return_sequences=False, activation='relu',
                       kernel_regularizer=l2(l2_reg),
                       recurrent_regularizer=l2(l2_reg),
                       dropout=0.2,
                       recurrent_dropout=0.2))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        for units in dense_units:
            model.add(Dense(units, activation='relu',
                            kernel_regularizer=l2(l2_reg)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        model.add(Dense(num_classes, activation='softmax'))

        return model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=150):
        """Train with optimized settings"""

        num_classes = y_train.shape[1]
        print(f"\nTraining ULTRA-OPTIMIZED model with {num_classes} classes")

        model = self.build_model(num_classes, model_size='small')

        #lower initial learning rate
        optimizer = Adam(learning_rate=0.0005)  # Lower than default 0.001

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

        model.summary()

        log_dir = os.path.join('logs', 'ultra_optimized_logs')
        os.makedirs(log_dir, exist_ok=True)

        callbacks = [
            TensorBoard(log_dir=log_dir),
            ModelCheckpoint('asl_ultra_model.h5',
                            save_best_only=True,
                            monitor='val_categorical_accuracy',
                            verbose=1,
                            mode='max'),
            #reduce learning rate when validation loss stays stagnant
            ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=8,  #
                              min_lr=0.000001,
                              verbose=1),
            # early stop at 40 epochs of no improvement
            EarlyStopping(monitor='val_categorical_accuracy',
                          patience=40,
                          restore_best_weights=True,
                          verbose=1,
                          mode='max')
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )

        return model, history

    def prepare_training_data(self, gloss_subset=None):
        """Load and prepare data with MAXIMUM augmentation"""

        X_train, y_train_labels = self.load_processed_data('train', gloss_subset=gloss_subset)
        X_val, y_val_labels = self.load_processed_data('val', gloss_subset=gloss_subset)
        X_test, y_test_labels = self.load_processed_data('test', gloss_subset=gloss_subset)

        unique_glosses = sorted(list(set(y_train_labels)))
        gloss_to_int = {gloss: idx for idx, gloss in enumerate(unique_glosses)}

        y_train = np.array([gloss_to_int[label] for label in y_train_labels])
        y_val = np.array([gloss_to_int[label] for label in y_val_labels])
        y_test = np.array([gloss_to_int[label] for label in y_test_labels])

        # max aug at 20x
        print("\n" + "=" * 60)
        print("APPLYING MAXIMUM AUGMENTATION (20x)")
        print("=" * 60)
        from data_augmentor import augment_training_data
        X_train, y_train = augment_training_data(X_train, y_train, 20)

        num_classes = len(unique_glosses)
        y_train = to_categorical(y_train, num_classes)
        y_val = to_categorical(y_val, num_classes)
        y_test = to_categorical(y_test, num_classes)

        print(f"\nDataset shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_val: {X_val.shape}")
        print(f"X_test: {X_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test, unique_glosses, gloss_to_int


def train_ultra_optimized(gloss_subset):
    """Train with maximum optimizations"""

    trainer = ASLCitizenTrainer()

    X_train, y_train, X_val, y_val, X_test, y_test, glosses, mapping = \
        trainer.prepare_training_data(gloss_subset)

    model, history = trainer.train_model(X_train, y_train, X_val, y_val, epochs=150)

    #eval
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    test_acc = accuracy_score(y_true_classes, y_pred_classes)

    # best accuracies
    top5_acc = sum(1 for i in range(len(y_pred))
                   if y_true_classes[i] in np.argsort(y_pred[i])[-5:]) / len(y_pred)

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Top-5 Accuracy: {top5_acc:.4f} ({top5_acc * 100:.2f}%)")

    # Save
    import json
    with open('asl_ultra_glosses.json', 'w') as f:
        json.dump(mapping, f, indent=2)

    return model, glosses, mapping


if __name__ == "__main__":
    import json

    with open('recommended_classes_top250.json', 'r') as f:
        top_classes = json.load(f)[:50]

    model, glosses, mapping = train_ultra_optimized(top_classes)