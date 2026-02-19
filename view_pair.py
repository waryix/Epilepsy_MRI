import os
import nibabel as nib
import matplotlib.pyplot as plt

# Paths to your dataset folders
img_dir = r"C:\Users\91954\Downloads\HFH\Train"
lbl_dir = r"C:\Users\91954\Downloads\HFH\Train\Labels"

# List all MRI images (assuming they have .img extension)
img_files = [f for f in os.listdir(img_dir) if f.endswith(".img")]

# Number of pairs to display
num_pairs = 5
img_files = img_files[:num_pairs]

# Create a figure with rows = number of pairs, columns = 2 (MRI + mask)
plt.figure(figsize=(10, 4 * num_pairs))

for i, img_file in enumerate(img_files):
    img_path = os.path.join(img_dir, img_file)
    
    # Assuming the mask file has the same naming pattern with "_Hipp_Labels"
    lbl_file = img_file.replace(".img", "_Hipp_Labels.img")
    lbl_path = os.path.join(lbl_dir, lbl_file)
    
    img = nib.load(img_path).get_fdata()
    lbl = nib.load(lbl_path).get_fdata()
    
    slice_idx = img.shape[2] // 2  # middle slice
    
    # MRI subplot
    plt.subplot(num_pairs, 2, i*2 + 1)
    plt.imshow(img[:, :, slice_idx], cmap='gray')
    plt.title(f"MRI: {img_file}")
    plt.axis("off")
    
    # Mask subplot
    plt.subplot(num_pairs, 2, i*2 + 2)
    plt.imshow(lbl[:, :, slice_idx], cmap='jet')
    plt.title(f"Mask: {lbl_file}")
    plt.axis("off")

plt.tight_layout()
plt.show()
