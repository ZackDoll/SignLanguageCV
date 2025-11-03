import json
import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import urllib.request
from pathlib import Path

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


class MSASLDataLoader:
    """Loader for MSASL dataset"""

    def __init__(self, dataset_dir='MSASL_data', processed_dir='MP_Data_MSASL'):
        self.dataset_dir = dataset_dir
        self.processed_dir = processed_dir
        self.mp_holistic = mp_holistic
        self.synonym_map = None

    def load_synonyms(self):
        """synonym mappings (2d array)
        Format: [["ticket", "give ticket"], ["get", "receive"], ...]
        """
        synonym_path = os.path.join(self.dataset_dir, 'MSASL_synonym.json')
        try:
            with open(synonym_path, 'r') as f:
                synonyms = json.load(f)

            # Create a mapping from any synonym to the first (canonical) term
            synonym_map = {}
            for syn_group in synonyms:
                canonical = syn_group[0]  # First term is canonical
                for term in syn_group:
                    synonym_map[term] = canonical

            return synonym_map
        except Exception as e:
            print(f"Could not load synonyms: {e}")
            return {}

    def load_json_files(self):
        """Loads JSON files from MS-ASL dataset"""
        train_path = os.path.join(self.dataset_dir, 'MSASL_train.json')
        test_path = os.path.join(self.dataset_dir, 'MSASL_test.json')
        val_path = os.path.join(self.dataset_dir, 'MSASL_val.json')
        classes_path = os.path.join(self.dataset_dir, 'MSASL_classes.json')

        with open(train_path, 'r') as f:
            train_data = json.load(f)  # Array of sample dicts
        with open(test_path, 'r') as f:
            test_data = json.load(f)  # Array of sample dicts
        with open(val_path, 'r') as f:
            val_data = json.load(f)  # Array of sample dicts
        with open(classes_path, 'r') as f:
            classes = json.load(f)  # Array like ["ticket", "nice", "teacher", ...]

        return train_data, test_data, val_data, classes

    def download_video(self, url, output_path):
        """Downloads YT video"""
        try:
            # For YouTube videos, we need yt-dlp
            try:
                import yt_dlp

                ydl_opts = {
                    'format': 'best[ext=mp4]',
                    'outtmpl': output_path,
                    'quiet': True,
                    'no_warnings': True,
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                return True

            except ImportError:
                print("ERROR: Please install yt-dlp: pip install yt-dlp")
                return False

        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False

    def extract_clip(self, video_path, start_time, end_time, output_path):
        """Extract clip from video between start_time and end_time"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # set start to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # define videowriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps,
                              (int(cap.get(3)), int(cap.get(4))))

        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame += 1

        cap.release()
        out.release()
        return True

    def crop_video_to_box(self, frame, box, width, height):
        """crops video to bounding box to normalize data"""
        y0, x0, y1, x1 = box
        # Denormalize coordinates
        x0 = int(x0 * width)
        x1 = int(x1 * width)
        y0 = int(y0 * height)
        y1 = int(y1 * height)

        return frame[y0:y1, x0:x1]

    def mediapipe_detection(self, image, model):
        """Detects parts of body in image"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def extract_keypoints(self, results):
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

    def process_video_to_keypoints(self, video_path, box, width, height,
                                   num_frames=30, use_box=True):
        """Process video and extract keypoints for fixed number of frames (30fps rn)"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        keypoints_sequence = []

        with mp_holistic.Holistic(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as holistic:
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    # If frame not available, use zeros
                    keypoints_sequence.append(np.zeros(1662))
                    continue

                #crop to bounding box
                if use_box and box:
                    frame = self.crop_video_to_box(frame, box, width, height)

                # Resize to standard size
                frame = cv2.resize(frame, (640, 480))

                # Extract keypoints
                _, results = self.mediapipe_detection(frame, holistic)
                keypoints = self.extract_keypoints(results)
                keypoints_sequence.append(keypoints)

        cap.release()

        # makes sure num_frames is accurate
        while len(keypoints_sequence) < num_frames:
            keypoints_sequence.append(np.zeros(1662))

        return np.array(keypoints_sequence[:num_frames])

    def process_dataset_sample(self, sample, sample_idx, split='train',
                               num_frames=30, use_box=True):
        """Process a single sample from the dataset
        Sample format: {
            'url': str,
            'start_time': float,
            'end_time': float,
            'label': int,
            'text': str,
            'box': [y0, x0, y1, x1],
            'width': float,
            'height': float,
            'fps': float,
            ...
        }
        """
        url = sample['url']
        start_time = sample['start_time']
        end_time = sample['end_time']
        label = sample['label']
        text = sample['text']
        box = sample.get('box', None)
        width = sample.get('width', 640)
        height = sample.get('height', 480)

        # creates directory
        label_dir = os.path.join(self.processed_dir, split, str(label))
        os.makedirs(label_dir, exist_ok=True)

        # dont process duplicates
        keypoint_path = os.path.join(label_dir, f"{sample_idx}.npy")
        if os.path.exists(keypoint_path):
            return True

        # ensures http protocol
        if not url.startswith('http'):
            url = 'https://' + url

        # downloader
        video_dir = os.path.join(self.dataset_dir, 'videos', split)
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"{sample_idx}.mp4")

        if not os.path.exists(video_path):
            if not self.download_video(url, video_path):
                return False

        # process to np keypoints and save them
        try:
            keypoints = self.process_video_to_keypoints(
                video_path, box, width, height, num_frames, use_box
            )

            np.save(keypoint_path, keypoints)

            # delete video to save space if needed
            # os.remove(video_path)

            return True
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            return False

    def process_full_dataset(self, num_frames=30, use_box=True,
                             limit_per_split=None, skip_existing=True):
        """Process entire MS-ASL dataset

        Args:
            num_frames: Number of frames to extract per video
            use_box: Whether to crop to bounding box
            limit_per_split: Limit number of samples per split (for testing)
            skip_existing: Skip already processed samples
        """
        # CREATE DIRECTORIES FIRST
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(os.path.join(self.processed_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.processed_dir, 'test'), exist_ok=True)
        os.makedirs(os.path.join(self.processed_dir, 'val'), exist_ok=True)

        train_data, test_data, val_data, classes = self.load_json_files()
        self.synonym_map = self.load_synonyms()

        print(f"Total classes: {len(classes)}")
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")
        print(f"Val samples: {len(val_data)}")
        print(f"Synonyms loaded: {len(self.synonym_map)} mappings")

        # process the splits
        for split_name, split_data in [('train', train_data),
                                       ('test', test_data),
                                       ('val', val_data)]:
            print(f"\nProcessing {split_name} split...")

            if limit_per_split:
                split_data = split_data[:limit_per_split]

            successful = 0
            failed = 0
            skipped = 0

            for idx, sample in enumerate(tqdm(split_data, desc=f"{split_name}")):
                # dont repeat data
                label = sample['label']
                label_dir = os.path.join(self.processed_dir, split_name, str(label))
                keypoint_path = os.path.join(label_dir, f"{idx}.npy")

                if skip_existing and os.path.exists(keypoint_path):
                    skipped += 1
                    continue

                if self.process_dataset_sample(sample, idx, split_name,
                                               num_frames, use_box):
                    successful += 1
                else:
                    failed += 1

            print(f"Successfully processed: {successful}/{len(split_data)}")
            print(f"Failed: {failed}")
            print(f"Skipped (already exists): {skipped}")

if __name__ == "__main__":
    loader = MSASLDataLoader(
        dataset_dir='MSASL_data',
        processed_dir='MP_Data_MSASL'
    )

    #process the set
    loader.process_full_dataset(
        num_frames=30,
        use_box=True,
        limit_per_split=1000, # Remove this to process full dataset (defaults to NoneType)

    )