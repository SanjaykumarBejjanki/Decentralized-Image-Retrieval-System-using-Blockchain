from keras.applications import InceptionV3
from keras.models import Model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import numpy as np
from PIL import Image
import cv2
import os

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extraction_model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed10').output)
    
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    return preprocessed_img

path = "stex-512"
X = []
Y = []

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        img = Image.open(root+'/'+directory[j])
        img = np.array(img)
        cv2.imwrite("test.jpg", img)
        image_path = 'test.jpg'
        processed_image = preprocess_image(image_path)
        image_features = feature_extraction_model.predict(processed_image)
        image_features = image_features.ravel()
        X.append(image_features)
        Y.append(directory[j])
        print(j)

X = np.asarray(X)
Y = np.asarray(Y)

np.save('model/stex_google_X',X)
np.save('model/stex_google_Y',Y)

X = np.load('model/stex_google_X.npy')
Y = np.load('model/stex_google_Y.npy')
        
print(X)
print(Y)
print(X.shape)
print(Y.shape)




