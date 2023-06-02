import numpy as np
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import os

# Photos are of size 48x48
photosize = 48

# Path of the Dataset
path = 'C:/Users/sidar/images/'

# Creating a figure using matplotlib
plt.figure(0, figsize=(12, 20))
cpt = 0

for emotion in os.listdir(path + "train"):
	#Taking 5 images of each Emotion
	for i in range(5):
		cpt = cpt+1
		plt.subplot(7, 5, cpt)
		img = load_img(path + "train/" + emotion + "/" + os.listdir(path + "train/" + emotion)[i], target_size=(photosize, photosize))
		plt.imshow(img, cmap = "gray")
#To show the images using matplotlib
plt.show()
