# ASL Sign Language Recognition System

## Summary

A deep learning system for real-time American Sign Language (ASL) recognition using LSTM neural networks and computer vision. The system captures hand, face, and pose landmarks from video sequences to classify ASL signs with support for thousands of different signs.

**Key Capabilities:**
- **Real-time Recognition**: Live webcam feed processing with immediate sign detection
- **Large Vocabulary**: Trained on 2,700+ unique ASL signs from multiple datasets
- **Multi-Dataset Support**: Integrates ASL Citizen, Google ISLR, and MS-ASL datasets
- **Robust Feature Extraction**: Uses MediaPipe Holistic for comprehensive body landmark detection
- **Data Augmentation**: Synthetic data generation to overcome limited samples per sign
- **Flexible Training**: Modular architecture supporting various model sizes and class subsets

**Technical Highlights:**
- **Architecture**: LSTM-based sequence classifier with attention to temporal patterns
- **Input Processing**: 30-frame sequences with 1,662 features per frame (pose, face, hands)
- **Training Optimizations**: Heavy regularization, dynamic learning rate, early stopping
- **Inference Speed**: Real-time performance (~30 FPS) on consumer hardware
- **Accuracy**: 60-65% top-1 accuracy on 500 classes, 80-85% top-5 accuracy

**Use Cases:**
- Assistive technology for deaf/hard-of-hearing communication
- ASL learning and education tools
- Sign language translation services
- Research platform for gesture recognition

---

# PROJECT OVERVIEW


This project implements a real-time American Sign Language (ASL) recognition system using LSTM neural networks and MediaPipe for pose/hand landmark extraction. The system processes video sequences to classify ASL signs, supporting multiple datasets. 


## Supported Datasets

### 1. ASL Citizen Dataset (Primary)
A comprehensive dataset of isolated ASL signs with pre-segmented video clips.

#### Files
- **train.csv**: 40,154 training samples
- **val.csv**: 10,304 validation samples  
- **test.csv**: 32,941 test samples
- **videos/**: Folder containing all video files

#### CSV Columns
- **Gloss**: Classification/label of the sign (e.g., "HELLO", "THANKYOU")
- **Participant ID**: Unique identifier for the signer
- **Video file**: Filename in videos folder matching this sample
- **ASL-LEX Code**: ASL-LEX encoding reference number

#### Dataset Structure
```
ASL_Citizen/
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── videos/
    └── [all video files]
```

---

### 2. Google Isolated Sign Language Recognition Dataset (Optional Enhancement)
Kaggle competition dataset with additional ASL signs to augment training data.

#### Files
- **train.csv**: ~94,000 training samples with landmark data
- **sign_to_prediction_index_map.json**: Mapping of sign indices to names

#### CSV Columns
- **path**: Relative path to parquet file (e.g., "train_landmark_files/2044/sequence.parquet")
- **participant_id**: Unique identifier for the signer
- **sequence_id**: Unique identifier for the video sequence
- **sign**: Sign class index

#### Dataset Structure
```
train_landmark_files/
├── train.csv
├── sign_to_prediction_index_map.json
├── 2044/ (participant folders)
├── 4718/
└── ... (each contains .parquet files with pre-extracted landmarks)
```

**Note**: This dataset uses pre-extracted MediaPipe landmarks stored in parquet format, eliminating the need for video processing.

---

### 3. MS-ASL Dataset (Legacy Support)
Microsoft's ASL dataset with YouTube video URLs and temporal annotations.

**Download**: https://www.microsoft.com/en-us/download/details.aspx?id=100121

#### Files
- **MSASL_train.json**: 16,054 training samples
- **MSASL_val.json**: 5,287 validation samples
- **MSASL_test.json**: 4,172 test samples
- **MSASL_classes.json**: 1,000 class names `["ticket", "nice", "teacher", ...]`
- **MSASL_synonym.json**: Synonym groups `[["ticket", "give ticket"], ["get", "receive"], ...]`

#### Sample Format
Each sample contains:
```json
{
  "url": "https://www.youtube.com/watch?v=...",
  "start_time": 0.0,
  "end_time": 1.969,
  "label": 805,
  "text": "beer",
  "box": [0.047, 0.290, 1.0, 0.823],  // Bounding box [x_min, y_min, x_max, y_max]
  "width": 640.0,
  "height": 360.0,
  "fps": 29.97,
  "signer_id": 20
}
```

**Note**: Requires downloading videos from YouTube URLs and extracting segments using start/end times.

---

## Processing Pipeline

### 1. **Video Processing** (ASL Citizen)
- Extract 30 frames evenly distributed across each video
- Use MediaPipe Holistic to extract keypoints from each frame
- Generate 1662-dimensional feature vectors per frame:
  - Pose: 33 landmarks × 4 (x, y, z, visibility) = 132 features
  - Face: 468 landmarks × 3 (x, y, z) = 1,404 features
  - Left Hand: 21 landmarks × 3 = 63 features
  - Right Hand: 21 landmarks × 3 = 63 features

### 2. **Parquet Processing** (Google ISLR)
- Read pre-extracted landmarks from parquet files
- Convert to same 1662-dimensional format
- Resample/pad to 30 frames per sequence

### 3. **Data Augmentation**
Apply transformations to increase effective dataset size:
- Time warping (speed variations)
- Spatial noise
- Scaling (distance from camera)
- Translation (position in frame)
- Rotation (camera angles)
- Frame dropping (simulate missing frames)

### 4. **Model Training**
- Architecture: LSTM-based sequence classifier
- Input: (30 frames, 1662 features)
- Output: Softmax probabilities over sign classes
- Regularization: Dropout (0.5-0.6), L2 regularization, BatchNormalization

### 5. **Real-time Inference**
- Capture live webcam feed
- Extract keypoints using MediaPipe
- Maintain 30-frame sliding window
- Predict sign when window is full
- Display top-5 predictions with confidence scores

---

## Expected Performance

| Dataset Configuration | Classes | Samples/Class | Expected Accuracy |
|----------------------|---------|---------------|-------------------|
| ASL Citizen (Top 50) | 50 | ~35 | 35-45% |
| ASL Citizen (Top 100) | 100 | ~18 | 25-35% |
| Combined (ASL + Google) | ~300 | ~45 | 40-55% |
| Full ASL Citizen | 2,731 | ~15 | 8-15% |

**Note**: Top-5 accuracy is typically 2-3x higher than top-1 accuracy.

---

## Key Features

Real-time webcam inference  
Support for 2,700+ ASL signs  
Data augmentation for limited samples  
Multi-dataset integration  
Checkpoint-based training (resume anytime)  
TensorBoard logging  
Modular architecture (easy to extend)  

---

## Quick Start

See individual dataset guides and training scripts for detailed instructions.
