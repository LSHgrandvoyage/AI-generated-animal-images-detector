import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

def load_images_from_folder(folder, size=(256, 256)):
    images = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, size)
        images.append(img)
    return images

def compute_fft_magnitude(img):
    # 2D FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    return magnitude

def compute_average_spectrum(images):
    if len(images) == 0:
        return None
    h, w = images[0].shape
    sum_mag = np.zeros((h, w), dtype=np.float32)

    for img in images:
        mag = compute_fft_magnitude(img)
        sum_mag += mag

    return sum_mag / len(images)

# ---- MAIN ----
real_dir = os.path.join(os.getenv("DATA_PATH"), 'train', 'real')
ai_dir   = os.path.join(os.getenv("DATA_PATH"), 'train', 'ai')

real_imgs = load_images_from_folder(real_dir)
ai_imgs   = load_images_from_folder(ai_dir)

print(f"Loaded {len(real_imgs)} real images")
print(f"Loaded {len(ai_imgs)} AI images")

real_spectrum = compute_average_spectrum(real_imgs)
ai_spectrum   = compute_average_spectrum(ai_imgs)

plt.figure(figsize=(14,6))

plt.subplot(1, 2, 1)
plt.title("Average Frequency Spectrum (Real)")
plt.imshow(np.log(real_spectrum + 1), cmap='inferno')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Average Frequency Spectrum (AI)")
plt.imshow(np.log(ai_spectrum + 1), cmap='inferno')
plt.colorbar()

plt.tight_layout()
plt.show()
