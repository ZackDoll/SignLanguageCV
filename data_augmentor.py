import numpy as np
import random


class DataAugmenter:
    """Augment keypoint sequences to increase training data"""

    def __init__(self):
        pass

    def time_warp(self, sequence, sigma=0.2):
        """Randomly speed up or slow down the sequence"""
        num_frames = len(sequence)
        #warp the time indices
        warp = np.random.normal(1.0, sigma, num_frames)
        warp = np.cumsum(warp)
        warp = (warp - warp[0]) / (warp[-1] - warp[0]) * (num_frames - 1)

        # new time points
        warped_sequence = np.zeros_like(sequence)
        for i in range(num_frames):
            idx = int(warp[i])
            if idx < num_frames - 1:
                #linear interpolating
                alpha = warp[i] - idx
                warped_sequence[i] = (1 - alpha) * sequence[idx] + alpha * sequence[idx + 1]
            else:
                warped_sequence[i] = sequence[-1]

        return warped_sequence

    def add_noise(self, sequence, noise_level=0.01):
        """Add Gaussian noise to keypoints"""
        noise = np.random.normal(0, noise_level, sequence.shape)
        return sequence + noise

    def scale(self, sequence, scale_range=(0.9, 1.1)):
        """Randomly scale the spatial coordinates"""
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])

        # scale the x and y coords
        scaled = sequence.copy()

        # Pose: indices 0-132 (33 landmarks × 4)
        for i in range(0, 132, 4):
            scaled[:, i:i + 2] *= scale_factor  # x, y only

        # Face: indices 132-1536 (468 landmarks × 3)
        for i in range(132, 1536, 3):
            scaled[:, i:i + 2] *= scale_factor  # x, y only

        # Left hand: indices 1536-1599 (21 landmarks × 3)
        for i in range(1536, 1599, 3):
            scaled[:, i:i + 2] *= scale_factor

        # Right hand: indices 1599-1662 (21 landmarks × 3)
        for i in range(1599, 1662, 3):
            scaled[:, i:i + 2] *= scale_factor

        return scaled

    def translate(self, sequence, translate_range=(-0.05, 0.05)):
        """Randomly translate the keypoints"""
        tx = np.random.uniform(translate_range[0], translate_range[1])
        ty = np.random.uniform(translate_range[0], translate_range[1])

        translated = sequence.copy()

        #pose
        for i in range(0, 132, 4):
            translated[:, i] += tx  # x
            translated[:, i + 1] += ty  # y

        #face tessalations
        for i in range(132, 1536, 3):
            translated[:, i] += tx
            translated[:, i + 1] += ty

        #lh
        for i in range(1536, 1599, 3):
            translated[:, i] += tx
            translated[:, i + 1] += ty

        #rh
        for i in range(1599, 1662, 3):
            translated[:, i] += tx
            translated[:, i + 1] += ty

        return translated

    def rotate(self, sequence, angle_range=(-10, 10)):
        """Rotate keypoints around center"""
        angle = np.random.uniform(angle_range[0], angle_range[1])
        angle_rad = np.deg2rad(angle)

        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        rotated = sequence.copy()

        # rotate x and y coords
        def rotate_xy(idx_x, idx_y):
            x = sequence[:, idx_x]
            y = sequence[:, idx_y]
            rotated[:, idx_x] = x * cos_a - y * sin_a
            rotated[:, idx_y] = x * sin_a + y * cos_a

        # pose
        for i in range(0, 132, 4):
            rotate_xy(i, i + 1)

        # face
        for i in range(132, 1536, 3):
            rotate_xy(i, i + 1)

        # lh + rh
        for i in range(1536, 1599, 3):
            rotate_xy(i, i + 1)
        for i in range(1599, 1662, 3):
            rotate_xy(i, i + 1)

        return rotated

    def drop_frames(self, sequence, drop_prob=0.1):
        """Randomly drop and interpolate frames"""
        num_frames = len(sequence)
        keep_mask = np.random.random(num_frames) > drop_prob

        # Ensure at least some frames remain
        if keep_mask.sum() < num_frames // 2:
            return sequence

        kept_indices = np.where(keep_mask)[0]
        dropped_sequence = sequence.copy()

        # interpret dropped frames
        for i in range(num_frames):
            if not keep_mask[i]:
                #find nearest kept frames
                before = kept_indices[kept_indices < i]
                after = kept_indices[kept_indices > i]

                if len(before) > 0 and len(after) > 0:
                    idx_before = before[-1]
                    idx_after = after[0]
                    alpha = (i - idx_before) / (idx_after - idx_before)
                    dropped_sequence[i] = (1 - alpha) * sequence[idx_before] + alpha * sequence[idx_after]

        return dropped_sequence

    def augment(self, sequence, num_augmentations=5):
        """Apply random augmentations to create multiple versions"""
        augmented = [sequence]  # Include original

        for _ in range(num_augmentations):
            aug_seq = sequence.copy()

            #apply each augmentation randomly
            if random.random() > 0.5:
                aug_seq = self.time_warp(aug_seq)
            if random.random() > 0.5:
                aug_seq = self.add_noise(aug_seq, noise_level=0.005)
            if random.random() > 0.5:
                aug_seq = self.scale(aug_seq)
            if random.random() > 0.5:
                aug_seq = self.translate(aug_seq)
            if random.random() > 0.7:
                aug_seq = self.rotate(aug_seq, angle_range=(-5, 5))
            if random.random() > 0.8:
                aug_seq = self.drop_frames(aug_seq, drop_prob=0.05)

            augmented.append(aug_seq)

        return augmented


def augment_training_data(X_train, y_train, augmentations_per_sample=5):
    """Augment entire training dataset"""
    augmenter = DataAugmenter()

    X_augmented = []
    y_augmented = []

    print(f"Augmenting {len(X_train)} training samples")
    print(f"{augmentations_per_sample} variations per sample")

    for i, (sequence, label) in enumerate(zip(X_train, y_train)):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(X_train)}")

        augmented_sequences = augmenter.augment(sequence, augmentations_per_sample)

        for aug_seq in augmented_sequences:
            X_augmented.append(aug_seq)
            y_augmented.append(label)


    return np.array(X_augmented), np.array(y_augmented)


if __name__ == "__main__":
    # test aug
    augmenter = DataAugmenter()

    #dummy sequence to augment
    test_sequence = np.random.randn(30, 1662) * 0.5 + 0.5

    augmented = augmenter.augment(test_sequence, num_augmentations=3)

    print(f"Original shape: {test_sequence.shape}")
    print(f"Generated {len(augmented)} variations")

    for i, aug in enumerate(augmented):
        print(f"  Variation {i}: shape {aug.shape}, mean {aug.mean():.4f}, std {aug.std():.4f}")