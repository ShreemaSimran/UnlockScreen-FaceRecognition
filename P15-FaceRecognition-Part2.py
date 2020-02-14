import cv2
import numpy as np
from os import listdir              #import operating system list directory
from os.path import isfile, join                 #OS libraries

data_path = 'E:/Data/'                            #'/' very important at the end
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]           #join the images

Training_Data, Labels = [], []               #split into instances using these variables

for i, files in enumerate(onlyfiles):             #enumerate contains both index value and value in i and files
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)       #read the grayscale images
    Training_Data.append(np.asarray(images, dtype=np.uint8))         #add training data
    Labels.append(i)                                        #add testing data

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()            #LocalBinaryPatternHistogram algorithm training model

model.train(np.asarray(Training_Data), np.asarray(Labels))               #input is matched with output equivalent to fit

print("Model Training Complete!!!!!")