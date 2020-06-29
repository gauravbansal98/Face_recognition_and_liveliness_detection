import face_recognition
import os
import json
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np

# a = json.load(emb)
# print(a['gaurav'])
# emb.close()
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def find_boxes(faces):
    boxes = []
    for result in faces:
        if result['confidence'] > .9:
            x, y, width, height = result['box']
            x_max = x + width
            y_max = y + height
            boxes.append((y, x+width, y+height, x))
    return boxes

detector = MTCNN()
embedding_dict = {}
emb_file = open('embeddings.json', 'w')
for folder in os.listdir('images'):
    embeddings = []
    for image in os.listdir(os.path.join('images', folder)):
        img = cv2.imread(os.path.join('images', folder, image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img)
        boxes = find_boxes(faces)
        if(len(boxes) != 1):
            print("Number of faces found in image {} is incorrect, please replace the file".format(os.path.join('images', folder, images)))
            continue
        embedding = face_recognition.face_encodings(img, boxes, num_jitters = 2)
        embeddings.append(embedding[0])
    embedding_dict[folder] = np.array(embeddings)
json.dump(embedding_dict, emb_file, cls=NumpyEncoder)
emb_file.close()