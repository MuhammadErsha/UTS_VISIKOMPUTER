# UTS_VISIKOMPUTER
import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_with_patch_kmeans(image_path, patch_size=(50, 50), k_clusters=2):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Size info
    h, w, c = img.shape
    segmented_img = np.zeros((h, w, c), dtype=np.uint8)

    # Loop over patches
    for y in range(0, h, patch_size[1]):
        for x in range(0, w, patch_size[0]):
            # Extract patch
            patch = img[y:y+patch_size[1], x:x+patch_size[0]]
            if patch.size == 0:
                continue

            # Reshape patch to (n_samples, n_features)
            pixel_values = patch.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)

            # Apply KMeans
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(pixel_values, k_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]
            segmented_patch = segmented_data.reshape(patch.shape)

            # Place segmented patch back into the result image
            segmented_img[y:y+patch_size[1], x:x+patch_size[0]] = segmented_patch

    return img, segmented_img

# Contoh penggunaan
image_path = 'path_ke_gambar_tbc.jpg'  # <-- Ganti dengan path file gambarmu
original, segmented = segment_with_patch_kmeans(image_path)

# Tampilkan hasil
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(original)
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Segmented Image (Patch-based KMeans)')
plt.imshow(segmented)
plt.axis('off')

plt.show()
