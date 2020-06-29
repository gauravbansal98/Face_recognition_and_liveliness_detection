import face_recognition
import os
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import json
import time

with open('embeddings.json', 'r') as embedding_file:
    embedding_dict = json.load(embedding_file)

with open('persons.txt', 'r') as f:
    persons = f.readlines()

def find_boxes(faces):
    boxes = []
    for result in faces:
        if result['confidence'] > .9:
            x, y, width, height = result['box']
            x_max = x + width
            y_max = y + height
            boxes.append((y, x+width, y+height, x))
    return boxes

cap = cv2.VideoCapture(0)

detector = MTCNN()

while True:
    ret, img = cap.read()
    if ret == True:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)
        boxes = find_boxes(faces)
        t = time.time()
        embeddings = face_recognition.face_encodings(rgb, boxes, num_jitters = 1)    
        for i, embedding in enumerate(embeddings):
            matches = []
            for person in persons:
                match = face_recognition.compare_faces(embedding_dict[person.rstrip()], embedding, tolerance = .55)
                matches.append(sum(match))
            cv2.rectangle(img, (boxes[i][3], boxes[i][0]), (boxes[i][1], boxes[i][2]), (255, 0, 0), 2)
            max_idx = matches.index(max(matches))
            if(matches[max_idx] > .5*len(embedding_dict[persons[max_idx].rstrip()])):
                cv2.putText(img, persons[max_idx].rstrip(), (boxes[i][3], boxes[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            else:
                cv2.putText(img, "Unknown Person", (boxes[i][3], boxes[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        print(time.time()-t)
        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

cap.release()  
cv2.destroyAllWindows()