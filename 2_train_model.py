import cv2
import os
import numpy as np

dataset_path = "dataset/"
faces = []
labels = []
label_dict = {}
i = 0

for file in os.listdir(dataset_path):
    img = cv2.imread(os.path.join(dataset_path, file), 0)
    faces.append(img)
    labels.append(i)

    name = file.split("_")[0]
    label_dict[i] = name
    i += 1

model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, np.array(labels))

model.write("trained_model.yml")
np.save("labels.npy", label_dict)

print("Training Completed Successfully!")