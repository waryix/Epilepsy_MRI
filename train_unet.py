import os
import nibabel as nib
import numpy as np
import cv2
import tensorflow as tf
from unet_model import unet
import albumentations as A

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ===============================
# Paths to MRI and Labels
# ===============================
TRAIN_IMG_DIR = r"dataset/Train"
TRAIN_LBL_DIR = r"dataset/Train/Labels"

# ===============================
# Normalize each volume
# ===============================
def normalize(vol):
    return (vol - vol.min()) / (vol.max() - vol.min())

# ===============================
# Data Generator
# ===============================
class MRIDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_files, lbl_dir, batch_size=8, augment=None, shuffle=True):
        self.img_files = img_files
        self.lbl_dir = lbl_dir
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # number of batches per epoch
        total_slices = sum([nib.load(f).shape[2] for f in self.img_files])
        return int(np.ceil(total_slices / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.img_files)

    def __getitem__(self, index):
        X_batch = []
        Y_batch = []

        # Loop until batch is filled
        while len(X_batch) < self.batch_size:
            # Pick a random volume
            vol_path = np.random.choice(self.img_files)
            fname = os.path.basename(vol_path)
            lbl_name = fname.replace(".img", "_Hipp_Labels.img")
            lbl_path = os.path.join(self.lbl_dir, lbl_name)

            if not os.path.exists(lbl_path):
                continue

            vol = normalize(nib.load(vol_path).get_fdata())
            lbl = nib.load(lbl_path).get_fdata()

            # Pick a random slice from this volume
            slice_idx = np.random.randint(0, vol.shape[2])
            im = cv2.resize(vol[:, :, slice_idx], (256, 256))
            m  = cv2.resize(lbl[:, :, slice_idx], (256, 256))
            m = (m > 0.5).astype(np.float32)

            # Data augmentation
            if self.augment:
                augmented = self.augment(image=im, mask=m)
                im = augmented['image']
                m  = augmented['mask']

            X_batch.append(im[..., np.newaxis])
            Y_batch.append(m[..., np.newaxis])

        return np.array(X_batch, dtype=np.float32), np.array(Y_batch, dtype=np.float32)

# ===============================
# Augmentation using Albumentations
# ===============================
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(p=0.3)
])

# ===============================
# Get all MRI image paths
# ===============================
all_images = [os.path.join(TRAIN_IMG_DIR, f) 
              for f in os.listdir(TRAIN_IMG_DIR) 
              if f.endswith(".img") and "Hipp" not in f]

# Shuffle & split into train/val volumes
np.random.shuffle(all_images)
split_idx = int(len(all_images) * 0.8)
train_files = all_images[:split_idx]
val_files   = all_images[split_idx:]

# ===============================
# Create Generators
# ===============================
train_gen = MRIDataGenerator(train_files, TRAIN_LBL_DIR, batch_size=8, augment=augmentations)
val_gen   = MRIDataGenerator(val_files, TRAIN_LBL_DIR, batch_size=8, augment=None, shuffle=False)

# ===============================
# Load U-Net model
# ===============================
model = unet(input_size=(256, 256, 1))

# ===============================
# Callbacks
# ===============================
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_hippo_unet.h5", monitor="val_dice_coef", save_best_only=True, mode="max", verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_dice_coef", patience=15, mode="max", restore_best_weights=True, verbose=1
)

# ===============================
# Train the model
# ===============================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=100,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# ===============================
# Save final model
# ===============================
model.save("final_hippo_unet.h5")
print("✅ Model training completed and saved!")
