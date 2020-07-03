import face_recognition
import os
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import json

class Face_recognition():
    def __init__(self):
        with open('embeddings.json', 'r') as embedding_file:
            self.embedding_dict = json.load(embedding_file)

        with open('persons.txt', 'r') as f:
            self.persons = f.readlines()

        self.detector = MTCNN()

    def find_boxes(self, faces):
        boxes = []
        for result in faces:
            if result['confidence'] > .9:
                x, y, width, height = result['box']
                boxes.append((y, x+width, y+height, x))
        return boxes

    def recognize(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb)
        boxes = self.find_boxes(faces)
        embeddings = face_recognition.face_encodings(rgb, boxes, num_jitters = 1)    
        for i, embedding in enumerate(embeddings):
            matches = []
            for person in self.persons:
                match = face_recognition.compare_faces(self.embedding_dict[person.rstrip()], embedding, tolerance = .55)
                matches.append(sum(match))
            cv2.rectangle(img, (boxes[i][3], boxes[i][0]), (boxes[i][1], boxes[i][2]), (255, 0, 0), 2)
            max_idx = matches.index(max(matches))
            if(matches[max_idx] > .5*len(self.embedding_dict[self.persons[max_idx].rstrip()])):
                cv2.putText(img, self.persons[max_idx].rstrip(), (boxes[i][3], boxes[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            else:
                cv2.putText(img, "Unknown Person", (boxes[i][3], boxes[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

        return img