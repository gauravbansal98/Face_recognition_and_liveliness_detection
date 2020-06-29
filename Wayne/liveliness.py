from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from mtcnn.mtcnn import MTCNN

class blink():
    def __init__(self):
        self.EYE_AR_THRESH = 0.23
        self.EYE_AR_CONSEC_FRAMES = 3

        self.COUNTER = 0

        self.shape_predictor = "shape_predictor_68_face_landmarks.dat"

        print("[INFO] loading facial landmark predictor...")
        self.detector = MTCNN()
        self.predictor = dlib.shape_predictor(self.shape_predictor)

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])
        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        # return the eye aspect ratio
        return ear

    def find_boxes(self, faces):
        boxes = []
        for result in faces:
            if result['confidence'] > .9:
                x, y, width, height = result['box']
                x_max = x + width
                y_max = y + height
                boxes.append([x, y, x_max, y_max])
        return boxes

    def detect_blink(self, frame):
        frame = imutils.resize(frame, width=450)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detect_faces(rgb)
        rects = self.find_boxes(faces)
        dlib_rects = []
        for rect in rects:
            dlib_rects.append(dlib.rectangle(rect[0],rect[1], rect[2], rect[3]))
        for rect in dlib_rects:
            shape = self.predictor(rgb, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            print(ear)
            if ear < self.EYE_AR_THRESH:
                self.COUNTER += 1
            else:
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    self.COUNTER = 0
                    return (frame, 1)
                self.COUNTER = 0
        if len(rects) == 0:
            return (frame, 2)
        return (frame, 0)