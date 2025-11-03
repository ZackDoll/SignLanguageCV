import json
import os
import time
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import hashlib
import yt_dlp
from yt_dlp.utils import DownloadError

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

        # video cache
        self.video_cache = {}
        self.cache_file = os.path.join(dataset_dir, 'video_cache.json')
        self.load_video_cache()

        # failed videos tracking
        self.failed_videos = set()
        self.failed_file = os.path.join(dataset_dir, 'failed_videos.json')
        self.load_failed_videos()

        # track rate limited videos
        self.rate_limited_videos = set()
        self.rate_limit_count = 0
        self.rate_limit_sleep_time = 60  # Start with 60 seconds

    def load_video_cache(self):
        """Load existing video download cache"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.video_cache = json.load(f)
            print(f"Loaded video cache with {len(self.video_cache)} entries")

    def save_video_cache(self):
        """Save video download cache"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.video_cache, f, indent=2)

    def load_failed_videos(self):
        """Load list of videos that failed to download"""
        if os.path.exists(self.failed_file):
            with open(self.failed_file, 'r') as f:
                self.failed_videos = set(json.load(f))
            print(f"Loaded {len(self.failed_videos)} known failed videos")

    def save_failed_videos(self):
        """Save list of failed videos"""
        with open(self.failed_file, 'w') as f:
            json.dump(list(self.failed_videos), f, indent=2)

    def get_video_hash(self, url):
        """Generate unique hash for video URL"""
        return hashlib.md5(url.encode()).hexdigest()

    def load_synonyms(self):
        """synonym mappings (2d array)
        Format: [["ticket", "give ticket"], ["get", "receive"], ...]
        """
        synonym_path = os.path.join(self.dataset_dir, 'MSASL_synonym.json')
        try:
            with open(synonym_path, 'r') as f:
                synonyms = json.load(f)

            # maps the synonym to the first term of synonym list
            synonym_map = {}
            for syn_group in synonyms:
                canonical = syn_group[0]  # "canon" term is easiest term to use
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

    def download_video(self, url, output_path, cookiefile='www.youtube.com_cookies.txt',
                       use_browser_cookies=False, max_retries=3, initial_backoff=5,
                       max_backoff=300):
        """
        Download with retries/backoff and optional cookie auth.
        Returns: (success: bool, is_rate_limited: bool, is_permanent_fail: bool)
        """
        attempt = 0
        backoff = initial_backoff

        # prepare opts
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'noprogress': True,
            'retries': 1,
        }

        if use_browser_cookies:
            ydl_opts['cookiesfrombrowser'] = ('opera',)
        elif cookiefile and os.path.exists(cookiefile):
            ydl_opts['cookiefile'] = cookiefile

        while attempt < max_retries:
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                return (True, False, False)  # success

            except DownloadError as e:
                msg = str(e).lower()
                attempt += 1

                # private or deleted errors
                if any(keyword in msg for keyword in ['private', 'unavailable', 'removed', 'deleted']):
                    # rate-limit check
                    if 'rate-limit' in msg or "isn't available" in msg or 'try again later' in msg:
                        return (False, True, False)  # rate-limited (temporary)
                    else:
                        return (False, False, True)  # permanent failure

                # youtube rate limits you
                if 'rate-limit' in msg or 'too many requests' in msg or 'exceeded' in msg:
                    return (False, True, False)  # rate-limited (temporary) (not successful, video exists, rate limited)

                # network error
                time.sleep(min(5 * attempt, 60))
                continue

            except Exception as e:
                attempt += 1
                time.sleep(min(5 * attempt, 60))
                continue

        # max retries, assume a temp failure
        return (False, True, False)

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

        # sample frames
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

                # crop to bounding box
                if use_box and box:
                    frame = self.crop_video_to_box(frame, box, width, height)

                # resize
                frame = cv2.resize(frame, (640, 480))

                # extract numpy keypoints
                _, results = self.mediapipe_detection(frame, holistic)
                keypoints = self.extract_keypoints(results)
                keypoints_sequence.append(keypoints)

        cap.release()

        # makes sure num_frames is accurate
        while len(keypoints_sequence) < num_frames:
            keypoints_sequence.append(np.zeros(1662))

        return np.array(keypoints_sequence[:num_frames])

    def process_dataset_sample(self, sample, sample_idx, split='train',
                               num_frames=30, use_box=True, delete_clip_after=False):
        """Process a single sample - with proper rate-limit handling"""
        url = sample['url']
        start_time = sample['start_time']
        end_time = sample['end_time']
        label = sample['label']
        text = sample['text']
        box = sample.get('box', None)
        width = sample.get('width', 640)
        height = sample.get('height', 480)

        # creates directory for this label
        label_dir = os.path.join(self.processed_dir, split, str(label))
        os.makedirs(label_dir, exist_ok=True)

        # check if already processed
        keypoint_path = os.path.join(label_dir, f"{sample_idx}.npy")
        if os.path.exists(keypoint_path):
            return True

        #make sure its http protocoled
        if not url.startswith('http'):
            url = 'https://' + url

        # Check if permanently failed (don't retry)
        if url in self.failed_videos:
            return False

        # check if rate limited (retry)
        if url in self.rate_limited_videos:
            return False

        # full video download
        video_hash = self.get_video_hash(url)
        full_video_dir = os.path.join(self.dataset_dir, 'videos', 'full')
        os.makedirs(full_video_dir, exist_ok=True)
        full_video_path = os.path.join(full_video_dir, f"{video_hash}.mp4")

        # checks the cache
        if url in self.video_cache and os.path.exists(full_video_path):
            full_video_path = self.video_cache[url]
        else:
            # download attempt
            success, is_rate_limited, is_permanent = self.download_video(url, full_video_path)

            if success:
                # add video to video cache
                self.video_cache[url] = full_video_path
                self.save_video_cache()

            elif is_rate_limited:
                # mark rate limited videos for retry
                self.rate_limited_videos.add(url)
                self.rate_limit_count += 1

                # sleeps for longer if we get multiple rate limited errors in a row
                if self.rate_limit_count >= 3:
                    print(f"\n⚠ Rate-limited! Pausing for {self.rate_limit_sleep_time}s...")
                    time.sleep(self.rate_limit_sleep_time)
                    # increases sleep time until un rate limited
                    self.rate_limit_sleep_time = min(self.rate_limit_sleep_time * 1.5, 600)
                    # clears the rate limited set and retries them
                    self.rate_limited_videos.clear()
                    self.rate_limit_count = 0

                return False

            elif is_permanent:
                # Permanent failure - mark as failed (video is private or deleted)
                video_id = url.split('=')[-1] if '=' in url else url[-11:]
                self.failed_videos.add(url)
                self.save_failed_videos()
                return False

            else:
                # whatever else is failing i guess
                return False

        # seperate clip and move it to clip_path
        clip_dir = os.path.join(self.dataset_dir, 'videos', 'clips', split)
        os.makedirs(clip_dir, exist_ok=True)
        clip_path = os.path.join(clip_dir, f"{sample_idx}.mp4")

        if not os.path.exists(clip_path):
            if not self.extract_clip(full_video_path, start_time, end_time, clip_path):
                return False

        # Process to keypoints
        try:
            keypoints = self.process_video_to_keypoints(
                clip_path, box, width, height, num_frames, use_box
            )
            np.save(keypoint_path, keypoints)

            if delete_clip_after and os.path.exists(clip_path):
                os.remove(clip_path)

            return True

        except Exception as e:
            return False

    def process_full_dataset(self, num_frames=30, use_box=True,
                             limit_per_split=None, skip_existing=True,
                             delete_clips_after=False):
        """Process entire MS-ASL dataset with rate-limit handling"""
        # CREATE DIRECTORIES
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

        # checks dupes
        print("\nAnalyzing video duplicates...")
        all_samples = train_data + test_data + val_data
        unique_urls = set(sample['url'] for sample in all_samples)
        duplicate_ratio = (len(all_samples) - len(unique_urls)) / len(all_samples) * 100
        print(f"Total samples: {len(all_samples)}")
        print(f"Unique videos: {len(unique_urls)}")
        print(f"Savings from deduplication: {duplicate_ratio:.1f}% fewer downloads")

        # retry logic for rate limited videos
        for split_name, split_data in [('train', train_data),
                                       ('test', test_data),
                                       ('val', val_data)]:
            print(f"\n{'=' * 60}")
            print(f"Processing {split_name} split")
            print(f"{'=' * 60}")

            if limit_per_split:
                split_data = split_data[:limit_per_split]

            successful = 0
            failed = 0
            skipped = 0
            rate_limited = 0

            for idx, sample in enumerate(tqdm(split_data, desc=f"{split_name}")):
                label = sample['label']
                label_dir = os.path.join(self.processed_dir, split_name, str(label))
                keypoint_path = os.path.join(label_dir, f"{idx}.npy")

                if skip_existing and os.path.exists(keypoint_path):
                    skipped += 1
                    continue

                # all rate limited videos go here
                url = sample.get('url', '')
                if not url.startswith('http'):
                    url = 'https://' + url

                was_rate_limited = url in self.rate_limited_videos

                if self.process_dataset_sample(sample, idx, split_name,
                                               num_frames, use_box, delete_clips_after):
                    successful += 1
                else:
                    if url in self.rate_limited_videos and not was_rate_limited:
                        rate_limited += 1
                    else:
                        failed += 1

            print(f"\n{split_name} Results:")
            print(f"  ✓ Successfully processed: {successful}/{len(split_data)}")
            print(f"  ✗ Permanently failed: {failed}")
            print(f"  ⏸ Rate-limited (will retry): {rate_limited}")
            print(f"  ⊘ Skipped (already exists): {skipped}")
            if (len(split_data) - skipped) > 0:
                success_rate = successful / (len(split_data) - skipped) * 100
                print(f"  Success rate: {success_rate:.1f}%")

        print(f"\n{'=' * 60}")
        print(f"Processing Complete!")
        print(f"{'=' * 60}")
        print(f"Permanently failed videos: {len(self.failed_videos)}")
        print(f"Rate-limited videos (retry later): {len(self.rate_limited_videos)}")

if __name__ == "__main__":
    loader = MSASLDataLoader(
        dataset_dir='MSASL_data',
        processed_dir='MP_Data_MSASL'
    )

    # process the set
    loader.process_full_dataset(
        num_frames=30,
        use_box=True,
        delete_clips_after=False,  # Set True to save disk space
        # limit_per_split=1000, # Remove this to process full dataset (defaults to NoneType)
    )