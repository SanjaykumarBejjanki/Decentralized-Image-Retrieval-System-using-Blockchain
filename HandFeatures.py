from scipy.ndimage import convolve
from skimage.feature import graycomatrix, graycoprops
import numpy as np
from PIL import Image
import cv2
import os

hog = cv2.HOGDescriptor()

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

def DDBTCfeatures(encoded_blocks):
    """Extracts features from DDBTC encoded data."""
    features = []
    for bitmap, min_level, max_level in encoded_blocks:
        bit_count = np.sum(bitmap)
        
        kernel = np.array([[-1, 1]])
        horizontal_transitions = np.sum(np.abs(convolve(bitmap, kernel, mode='constant')))
        vertical_transitions = np.sum(np.abs(convolve(bitmap, kernel.T, mode='constant')))
        
        features.extend([bit_count, horizontal_transitions, vertical_transitions, min_level, max_level])
    return features

path = "stex-512"
X = []
Y = []
distances = [5] # Offset
angles = [np.pi/2]  # Vertical Direction

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        img = Image.open(root+'/'+directory[j])
        img = np.array(img)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
        encoded_data = ddbtc_encode(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        ddbtc_features = DDBTCfeatures(encoded_data)
        ddbtc_features = np.asarray(ddbtc_features)
        ddbtc_features = ddbtc_features.ravel()
        h = hog.compute(img).ravel()
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue, saturation, intensity = cv2.split(hsv_image)
        hue = hue.flatten()
        saturation = saturation.flatten()
        intensity = intensity.flatten()
        glcm = graycomatrix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), distances=distances, angles=angles)
        glcm = graycoprops(glcm, 'correlation').ravel()
        hand_features = np.hstack([ddbtc_features, hue, saturation, intensity, glcm, h])
        hand_features = np.nan_to_num(hand_features, nan=0)
        X.append(hand_features)
        Y.append(directory[j])
        print(str(j)+" "+str(hand_features.shape))


X = np.asarray(X)
Y = np.asarray(Y)

np.save('model/stex_hand_X',X)
np.save('model/stex_hand_Y',Y)

X = np.load('model/stex_hand_X.npy')
Y = np.load('model/stex_hand_Y.npy')
        
print(X)
print(Y)
print(X.shape)
print(Y.shape)




