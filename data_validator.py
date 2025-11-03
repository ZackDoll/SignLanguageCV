import json
import os
from collections import Counter


class MSASLInspector:
    """Utility to inspect MS-ASL dataset structure and contents"""

    def __init__(self, dataset_dir='MSASL_data'):
        self.dataset_dir = dataset_dir

    def load_data(self):
        """Load all dataset files"""
        with open(os.path.join(self.dataset_dir, 'MSASL_train.json'), 'r') as f:
            train = json.load(f)
        with open(os.path.join(self.dataset_dir, 'MSASL_test.json'), 'r') as f:
            test = json.load(f)
        with open(os.path.join(self.dataset_dir, 'MSASL_val.json'), 'r') as f:
            val = json.load(f)
        with open(os.path.join(self.dataset_dir, 'MSASL_classes.json'), 'r') as f:
            classes = json.load(f)
        with open(os.path.join(self.dataset_dir, 'MSASL_synonym.json'), 'r') as f:
            synonyms = json.load(f)

        return train, test, val, classes, synonyms

    def print_dataset_stats(self):
        """Print overall dataset statistics"""
        train, test, val, classes, synonyms = self.load_data()

        print("=" * 60)
        print("MS-ASL DATASET STATISTICS")
        print("=" * 60)

        print(f"\nTotal Classes: {len(classes)}")
        print(f"Total Samples: {len(train) + len(test) + len(val)}")
        print(f"  - Train: {len(train)}")
        print(f"  - Test: {len(test)}")
        print(f"  - Validation: {len(val)}")
        print(f"\nSynonym Groups: {len(synonyms)}")

        # Class distribution
        print("\n" + "=" * 60)
        print("CLASS DISTRIBUTION")
        print("=" * 60)

        train_labels = [s['label'] for s in train]
        test_labels = [s['label'] for s in test]
        val_labels = [s['label'] for s in val]

        train_dist = Counter(train_labels)
        test_dist = Counter(test_labels)
        val_dist = Counter(val_labels)

        print(f"\nTrain - Most common classes:")
        for label, count in train_dist.most_common(10):
            print(f"  {classes[label]:20s} (label {label:3d}): {count:4d} samples")

        print(f"\nTrain - Least common classes:")
        for label, count in train_dist.most_common()[-10:]:
            print(f"  {classes[label]:20s} (label {label:3d}): {count:4d} samples")

        # Video sources
        print("\n" + "=" * 60)
        print("VIDEO SOURCES")
        print("=" * 60)

        youtube_count = sum(1 for s in train if 'youtube.com' in s['url'].lower())
        print(f"YouTube videos in train: {youtube_count}/{len(train)}")

        # Sample inspection
        print("\n" + "=" * 60)
        print("SAMPLE STRUCTURE (First training sample)")
        print("=" * 60)

        sample = train[0]
        for key, value in sample.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.4f}")
            else:
                print(f"  {key:20s}: {value}")

        # Duration statistics
        print("\n" + "=" * 60)
        print("VIDEO DURATION STATISTICS")
        print("=" * 60)

        durations = [(s['end_time'] - s['start_time']) for s in train]
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        print(f"Average clip duration: {avg_duration:.2f} seconds")
        print(f"Minimum clip duration: {min_duration:.2f} seconds")
        print(f"Maximum clip duration: {max_duration:.2f} seconds")

        # Synonym examples
        print("\n" + "=" * 60)
        print("SYNONYM EXAMPLES")
        print("=" * 60)

        for i, syn_group in enumerate(synonyms[:10]):
            print(f"{i + 1}. {' = '.join(syn_group)}")

    def find_samples_by_class(self, class_name, split='train', limit=5):
        """Find samples for a specific class"""
        train, test, val, classes, _ = self.load_data()

        # Find class index
        if class_name not in classes:
            print(f"Class '{class_name}' not found in dataset")
            print(f"Available classes: {classes[:20]}...")
            return

        class_idx = classes.index(class_name)

        # Get appropriate split
        if split == 'train':
            data = train
        elif split == 'test':
            data = test
        elif split == 'val':
            data = val
        else:
            print(f"Invalid split: {split}")
            return

        # Find samples
        samples = [s for s in data if s['label'] == class_idx]

        print(f"\nFound {len(samples)} samples for class '{class_name}' (label {class_idx}) in {split} split")
        print("\nSample URLs:")
        for i, sample in enumerate(samples[:limit]):
            print(f"\n{i + 1}. URL: {sample['url']}")
            print(f"   Time: {sample['start_time']:.2f}s - {sample['end_time']:.2f}s")
            print(f"   Duration: {sample['end_time'] - sample['start_time']:.2f}s")
            print(f"   Signer ID: {sample['signer_id']}")

    def check_label_consistency(self):
        """Check if labels match text descriptions"""
        train, test, val, classes, _ = self.load_data()

        print("\n" + "=" * 60)
        print("LABEL CONSISTENCY CHECK")
        print("=" * 60)

        all_data = train + test + val

        mismatches = []
        for sample in all_data[:100]:  # Check first 100
            label = sample['label']
            text = sample['text']
            expected_text = classes[label]

            if text != expected_text:
                mismatches.append((label, text, expected_text))

        if mismatches:
            print(f"\nFound {len(mismatches)} mismatches in first 100 samples:")
            for label, text, expected in mismatches[:10]:
                print(f"  Label {label}: '{text}' != '{expected}'")
        else:
            print("\nNo mismatches found - labels are consistent with text!")


if __name__ == "__main__":
    inspector = MSASLInspector('MSASL_data')

    # Print overall statistics
    inspector.print_dataset_stats()

    # Find samples for specific classes
    print("\n" + "=" * 60)
    inspector.find_samples_by_class('hello', split='train', limit=3)

    # Check label consistency
    inspector.check_label_consistency()