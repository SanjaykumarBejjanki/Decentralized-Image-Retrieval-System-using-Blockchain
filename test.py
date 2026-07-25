from PIL import Image
import cv2
import numpy as np
from scipy.ndimage import convolve

from skimage.feature import graycomatrix, graycoprops

hog = cv2.HOGDescriptor()

img = Image.open('stex-512/Bark.0000.pnm')
img = np.array(img)
img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
print(img.shape)
cv2.imwrite("test.jpg", img)

def dot_diffusion(block, mean):
    """Applies dot diffusion halftoning to a block."""
    error = 0
    bitmap = np.zeros_like(block, dtype=np.uint8)
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            value = block[i, j] + error
            if value >= mean:
                bitmap[i, j] = 1
                error = value - mean
            else:
                bitmap[i, j] = value - mean
    return bitmap

def ddbtc_encode(image, block_size=4):
    """Encodes an image using DDBTC."""
    height, width = image.shape
    encoded_blocks = []
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i + block_size, j:j + block_size]
            mean = np.mean(block)
            bitmap = dot_diffusion(block, mean)
            std_dev = np.std(block)
            
            max_level = mean + std_dev * np.sqrt(block_size*block_size/(block_size*block_size - np.sum(bitmap)))
            min_level = mean - std_dev * np.sqrt(np.sum(bitmap)/(block_size*block_size - np.sum(bitmap)))
            
            encoded_blocks.append((bitmap, min_level, max_level))
    return encoded_blocks

def extract_features(encoded_blocks):
    """Extracts features from DDBTC encoded data."""
    features = []
    for bitmap, min_level, max_level in encoded_blocks:
        bit_count = np.sum(bitmap)
        
        kernel = np.array([[-1, 1]])
        horizontal_transitions = np.sum(np.abs(convolve(bitmap, kernel, mode='constant')))
        vertical_transitions = np.sum(np.abs(convolve(bitmap, kernel.T, mode='constant')))
        
        features.extend([bit_count, horizontal_transitions, vertical_transitions, min_level, max_level])
    return features

# Example usage
image = np.random.randint(0, 256, size=(32, 32)).astype(np.uint8)
encoded_data = ddbtc_encode(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
features = extract_features(encoded_data)
features = np.asarray(features)
print(features.shape)
h = hog.compute(img).ravel()
print(h.shape)
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue, saturation, intensity = cv2.split(hsv_image)
hue = hue.flatten()
saturation = saturation.flatten()
intensity = intensity.flatten()
print(str(hue.shape)+" "+str(saturation.shape)+" "+str(intensity.shape))

# Generate GLCM
distances = [5] # Offset
angles = [np.pi/2]  # Vertical Direction
glcm = graycomatrix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), distances=distances, angles=angles,levels=255)
correlation = graycoprops(glcm, 'correlation')
print(correlation.shape)

