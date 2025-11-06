import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
from pathlib import Path



mp_holistic = mp.solutions.holistic


class SimpleASLDataLoader:
    """Loader for pre-clipped ASL dataset with CSV metadata"""

    def __init__(self, dataset_dir='ASL_dataset', processed_dir='Processed_Keypoints'):
        self.dataset_dir = dataset_dir
        self.processed_dir = processed_dir
        self.mp_holistic = mp_holistic

        # Expected structure
        self.videos_dir = os.path.join(dataset_dir, 'videos')
        self.splits_dir = os.path.join(dataset_dir, 'splits')

        os.makedirs(processed_dir, exist_ok=True)

    def load_metadata(self, split='train'):
        """Load CSV metadata for a split"""
        csv_path = os.path.join(self.splits_dir, f'{split}.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File {csv_path} does not exist")

        df = pd.read_csv(csv_path)

        print(f"Loaded {len(df)} samples from {split}.csv")
        print(f"Columns: {list(df.columns)}")
        print(f"Unique signs: {df['Gloss'].nunique()}")

        return df

    def mediapipe_detection(self, image, model):
        """Detects body parts in image"""
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

    def process_video_to_keypoints(self, video_path, num_frames=30):
        """Process video and extract keypoints for fixed number of frames"""
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            print(f"Warning: Video has 0 frames: {video_path}")
            cap.release()
            return None

        # samples frames based on num_frames
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        keypoints_sequence = []

        with mp_holistic.Holistic(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as holistic:
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    keypoints_sequence.append(np.zeros(1662))
                    continue

                # normalize
                frame = cv2.resize(frame, (640, 480))

                # keypoint extract
                _, results = self.mediapipe_detection(frame, holistic)
                keypoints = self.extract_keypoints(results)
                keypoints_sequence.append(keypoints)

        cap.release()

        # frame check
        while len(keypoints_sequence) < num_frames:
            keypoints_sequence.append(np.zeros(1662))

        return np.array(keypoints_sequence[:num_frames])

    def process_split(self, split='train', num_frames=30, skip_existing=True):
        """Process all videos in a split"""
        df = self.load_metadata(split)

        #output directory for split
        split_output_dir = os.path.join(self.processed_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)

        successful = 0
        failed = 0
        skipped = 0

        print(f"\nProcessing {split} split...")

        for index, row in tqdm(df.iterrows(), total=len(df), desc=split):
            #video info
            video_filename = row['Video file']
            gloss = row['Gloss']
            participant_id = row['Participant ID']

            gloss_dir = os.path.join(split_output_dir, gloss)
            os.makedirs(gloss_dir, exist_ok=True)

            # output file name (use row for uniqueness)
            output_filename = f"{index}_participant{participant_id}_{Path(video_filename).stem}.npy"
            output_path = os.path.join(gloss_dir, output_filename)

            #skip if already done (for stops and restarts)
            if skip_existing and os.path.exists(output_path):
                skipped += 1
                continue

            #video file location
            video_path = os.path.join(self.videos_dir, video_filename)

            keypoints = self.process_video_to_keypoints(video_path, num_frames)

            if keypoints is not None:
                #save keypoints
                """
                Get metadata from this func
                index = int(filename.split('_')[0]) 
                train_df = pd.read_csv('ASL_Citizen/splits/train.csv')
                metadata = train_df.iloc[index] #gets all columns
                TO ACCESS EACH PART
                metadata['Gloss'], etc.
                """
                np.save(output_path, keypoints)

                successful += 1
            else:
                failed += 1

        print(f"\n{split} Results:")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {failed}")
        print(f"Skipped (already exists): {skipped}")

        return {
            'successful': successful,
            'failed': failed,
            'skipped': skipped
        }

    def process_all_splits(self, num_frames=30, skip_existing=True):
        """Process train, test, and val splits"""
        print(f"{'=' * 60}")
        print(f"Processing All Splits")
        print(f"{'=' * 60}")

        for split in ['train', 'test', 'val']:
            self.process_split(split, num_frames, skip_existing)

        print(f"\n{'=' * 60}")
        print(f"Processing Complete!")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    loader = SimpleASLDataLoader(
        dataset_dir='ASL_Citizen',  #dataset folder
        processed_dir='Processed_Keypoints'
    )

    # process everything
    loader.process_all_splits(
        num_frames=30,
        skip_existing=True
    )
