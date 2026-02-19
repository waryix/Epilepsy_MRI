import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from skimage.transform import resize

# === PATHS ===
TEST_IMG_DIR = r"C:\Users\91954\Downloads\HFH\Test"
MODEL_PATH   = r"C:\Users\91954\epilepsy_MRI\hippo_unet_all.h5"

# === LOAD MODEL ===
model = load_model(MODEL_PATH)

IMG_HEIGHT = 256
IMG_WIDTH  = 256

def normalize(vol):
    return (vol - vol.min()) / (vol.max() - vol.min())

print("🔍 Starting HS detection on test dataset...\n")

for fname in os.listdir(TEST_IMG_DIR):
    if not fname.endswith(".img"):
        continue

    img_path = os.path.join(TEST_IMG_DIR, fname)
    print(f"Processing: {fname}")

    # Load MRI
    img = nib.load(img_path).get_fdata()
    img = np.squeeze(img)          # 🔥 FIX: force to 3D
    img = normalize(img)

    # === Predict mask for all slices ===
    pred_volume = np.zeros(img.shape)   # Now guaranteed 3D

    for i in range(img.shape[2]):
        slice_img = img[:, :, i]

        slice_resized = resize(
            slice_img,
            (IMG_HEIGHT, IMG_WIDTH),
            preserve_range=True,
            anti_aliasing=True
        )

        if slice_resized.max() != 0:
            slice_norm = slice_resized / slice_resized.max()
        else:
            slice_norm = slice_resized

        input_tensor = slice_norm[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)

        pred_mask = model.predict(input_tensor, verbose=0)[0, :, :, 0]

        # Resize back to original size
        pred_mask_orig = resize(
            pred_mask,
            (img.shape[0], img.shape[1]),
            preserve_range=True,
            anti_aliasing=True
        )

        pred_volume[:, :, i] = pred_mask_orig

    # === Binarize predicted mask ===
    pred_bin = pred_volume > 0.5

    # === Split left & right hippocampus ===
    mid_x = pred_bin.shape[0] // 2
    left_vol  = np.sum(pred_bin[:mid_x, :, :])
    right_vol = np.sum(pred_bin[mid_x:, :, :])

    # === Asymmetry Index ===
    AI = abs(left_vol - right_vol) / (left_vol + right_vol + 1e-8)

    # === HS Detection Rule ===
    hs_flag = "⚠ POSSIBLE HS" if AI > 0.15 else "Normal"

    print(f"   Left Volume : {left_vol}")
    print(f"   Right Volume: {right_vol}")
    print(f"   Asymmetry Index: {AI:.4f}")
    print(f"   Result: {hs_flag}\n")

    # === Visualization (middle slice) ===
    mid_slice = img.shape[2] // 2

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(img[:, :, mid_slice], cmap='gray')
    plt.title(f"MRI: {fname}")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(pred_bin[:, :, mid_slice], cmap='jet')
    plt.title("Predicted Hippocampus")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

print("✅ HS detection finished for all test images.")
