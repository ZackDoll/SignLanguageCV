# ASL Citizen Integration Guide

## Files
- **test.csv**: CSV File of 32,941 test samples
- **train.csv**: CSV File of 40,154 training samples
- **val.csv**: CSV File of 10,304 validation samples
- **videos Folder**: folder containing all videos in the training 
testing and validation data pieces

## Columns for CSV
 - Gross: Classification of sign
 - Participant ID: ID of the person signing
 - Video file: Name of the file in videos folder that matches to that data
 - ASL-VEX Code: ASL-VEX encoding of the file number

# MS-ASL Dataset Integration Guide

This guide explains how to use my sign language recognition system with the MS-ASL dataset.

UPDATED MODEL USES DIFFERENT DATA SET, SCROLL TO THAT IF USING UPDATED VERSION
## Download from this link:
https://www.microsoft.com/en-us/download/details.aspx?id=100121
## Dataset Structure

### Files
- **MSASL_train.json**: Array of 16,054 training samples
- **MSASL_test.json**: Array of 4,172 test samples  
- **MSASL_val.json**: Array of 5,287 validation samples
- **MSASL_classes.json**: Array of 1,000 class names: `["ticket", "nice", "teacher", ...]`
- **MSASL_synonym.json**: Array of synonym groups: `[["ticket", "give ticket"], ["get", "receive"], ...]`

### Sample Format
Each sample is a dictionary:
```json
{
  "url": "https://www.youtube.com/watch?v=...",
  "start_time": 0.0,
  "end_time": 1.969,
  "label": 805,
  "text": "beer",
  "box": [0.047, 0.290, 1.0, 0.823],
  "width": 640.0,
  "height": 360.0,
  "fps": 29.97,
  "signer_id": 20,
  ...
}
```

## Setup

### 1. Install Dependencies
```bash
pip install opencv-python mediapipe tensorflow scikit-learn # for the actual model things
pip install yt-dlp  # YT downloads
pip install tqdm    # Progress checking (for your own sanity)
```

### 2. Directory Structure
```
your_project/
├── MSASL_data/
│   ├── MSASL_train.json
│   ├── MSASL_test.json
│   ├── MSASL_val.json
│   ├── MSASL_classes.json
│   └── MSASL_synonym.json
├── MP_Data_MSASL/    #this is where all the keypoints for each gesture go
├── msasl_data_loader.py
├── msasl_trainer.py
├── msasl_inference.py
└── msasl_inspector.py
```

## Usage Workflow

### Step 1: Inspect the Dataset (Optional but Recommended)
```bash
python data_validator.py
```

This will show you:
- Dataset statistics
- Class distribution
- Sample structures
- Duration statistics
- Synonym mappings

### Step 2: Download and Preprocess Videos

**Start with a small test batch:**
```python
from MSASL_data_loader import MSASLDataLoader

loader = MSASLDataLoader(
    dataset_dir='MSASL_data',
    processed_dir='MP_Data_MSASL'
)

# SMALL test batch (10 samples per split)
loader.process_full_dataset(
    num_frames=30,
    use_box=True,
    limit_per_split=10,
    skip_existing=True
)
```

**Process the full dataset:**
```python
# This will take 15-20 hours depending on your internet speed (for me it took 20)
loader.process_full_dataset(
    num_frames=30,
    use_box=True,
    skip_existing=True  # Resume if interrupted
)
```

**What happens:**
1. Downloads videos from YouTube URLs
2. Extracts the relevant clip (start_time to end_time)
3. Crops to bounding box (optional)
4. Processes through MediaPipe to extract keypoints
5. Saves keypoints as `.npy` files organized by label

### Step 3: Train the Model

**Option A: Train on subset (for testing):**
```python
from data_train import train_msasl_model

# Train on first 50 classes with 50 samples per split
model, classes = train_msasl_model(
    class_subset=list(range(50)),
    max_samples=50,
    epochs=50
)
```

**Option B: Train on full dataset:**
```python
# Train on all 1000 (currently) classes
model, classes = train_msasl_model(
    epochs=100
)
```

**Output files:**
- `msasl_model.h5` - Trained model
- `msasl_classes_array.json` - Array format: `["ticket", "nice", ...]`
- `msasl_classes_dict.json` - Dict format: `{"0": "ticket", "1": "nice", ...}`

### Step 4: Run Real-Time Inference

```bash
python msasl_inference.py
```

Or customize:
```python
from msasl_inference import run_msasl_inference

run_msasl_inference(
    model_path='msasl_model.h5',
    classes_path='msasl_classes_dict.json',
    threshold=0.7,  # confidence threshold
    sequence_length=30
)
```

**Controls:**
- Press 'q' to quit
- The display shows:
  - Top 5 predictions with confidence bars
  - Current sentence at the top
  - MediaPipe landmarks overlaid on video

## Performance Considerations

### Storage Requirements
- **Videos**: ~100GB for full dataset (can be deleted after processing)
- **Keypoints**: ~5GB for processed keypoints
- **Model**: ~50MB

### Processing Time
- **Single video**: ~10-30 seconds
- **Full dataset**: ~10-20 hours (depends on internet speed and CPU)

### Training Time
- **50 classes**: ~1-2 hours on GPU
- **1000 classes**: ~10-20 hours on GPU

## IF POSSIBLE DO NOT USE CPU, MUCH SLOWER
## Tips and Best Practices

### 1. Start Small
Always test with a small subset first:
```python
limit_per_split=10  # Just 10 samples
class_subset=list(range(10))  # Just 10 classes
```

### 2. Resume Processing
The loader skips already processed videos:
```python
skip_existing=True  # Safe to re-run if interrupted
```

### 3. Monitor Progress
Both loader and trainer use progress bars to show status.

### 4. Handle Download Failures
Some YouTube videos may be unavailable. The loader continues with available videos.

### 5. GPU Acceleration
For training on 1000 classes, use a GPU:
```python
# Check if GPU is available
import tensorflow as tf
print("GPUs available:", tf.config.list_physical_devices('GPU'))
```

## Troubleshooting

### Problem: "yt-dlp not found"
```bash
pip install yt-dlp
```

### Problem: YouTube download fails
- Some videos may be removed or private
- Check if URL is accessible
- The loader will skip failed downloads

### Problem: Out of memory during training
- Reduce batch_size: `batch_size=16` instead of 32
- Use fewer classes: `class_subset=list(range(100))`
- Train on GPU if available

### Problem: Low accuracy
- Train for more epochs
- Increase training data per class
- Check if keypoints were extracted correctly
- Adjust confidence threshold

## Advanced Usage

### Custom Class Subset
```python
# Train on specific signs you care about (works, but honestly easier to
# self train those instead of using DB)
classes_of_interest = ['hello', 'thanks', 'please', 'sorry', 'help']
class_indices = [classes.index(c) for c in classes_of_interest]

model, classes = train_msasl_model(
    class_subset=class_indices,
    epochs=100
)
```

### Fine-tuning
```python
# Load pre-trained model and continue training
from tensorflow.keras.models import load_model

model = load_model('msasl_model.h5')
# Continue training with new data
```

### Custom MediaPipe Settings
```python
# In msasl_data_loader.py, modify:
mp_holistic.Holistic(
    min_detection_confidence=0.7,  # Higher = more strict
    min_tracking_confidence=0.7
)
```

## Citation
```
@inproceedings{joze2018ms,
  title={MS-ASL: A large-scale data set and benchmark for understanding American Sign Language},
  author={Joze, Hamid Reza Vaezi and Koller, Oscar},
  booktitle={BMVC},
  year={2019}
}
```
