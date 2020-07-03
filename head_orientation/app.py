import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator
import tensorflow as tf


cap = cv2.VideoCapture(0)

sess = tf.Session() #Launch the graph in a session.
my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object
detector = MTCNN()
# Load the weights from the configuration folders
my_head_pose_estimator.load_yaw_variables("cnn_cccdd_30k.tf")

class Head_pose():
    def __init__(self):
        self.straight = False
        self.left = False
        self.right = False

    def head_angle(self, image, face):
        x, y, width, height = face['box']
        x -= width/4
        y -= height/5
        height += height/3
        width += width/3
        image_part = np.copy(image[int(y):int(y+height), int(x):int(x+width), :])
        if(image_part.shape[0] == 0 or image_part.shape[1] == 0 or image_part.shape[2] == 0):
            return -1
        image_part = cv2.resize(image_part, (150, 150))
        yaw = my_head_pose_estimator.return_yaw(image_part)  # Evaluate the yaw angle using a CNN
        return yaw[0][0][0]

    def estimate_pose(self, img, width, height, percent_occupy):
        r = (0, 0, 255); g = (0, 255, 0)
        radius = 10; thickness = -1
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        center_coordinates = (int(img.shape[1]/2), int(img.shape[0]/2))
        top_left = (int(center_coordinates[0]-width/2), int(center_coordinates[1]-height/2))
        bottom_right = (int(center_coordinates[0]+width/2), int(center_coordinates[1]+height/2))
        cut_img = np.copy(img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :])
        faces = detector.detect_faces(rgb)
        img_area = width*height
        max_area = 0
        for i, face in enumerate(faces):
            if face['confidence'] > .9:
                width = face['box'][2]
                height = face['box'][3]
                area = width*height
                max_area = max(max_area, area)
                if area > percent_occupy*img_area and area < .95*img_area:
                    cv2.putText(img, "Pefect", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, g)
                    angle = self.head_angle(img, face)
                    print(angle)
                    if(angle > -20 and angle < 20):
                        self.straight = True
                    elif angle < -40:
                        self.left = True
                    elif angle > 50:
                        self.right = True
                    break
        if max_area < percent_occupy*img_area:
            cv2.putText(img, "come close", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, r)
        elif(max_area > .95*img_area):
            cv2.putText(img, "little back", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, r)
        if self.straight:
            cv2.circle(img, (20, 20), radius, g, thickness)
            cv2.putText(img, "Straight", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, g)
        else:
            cv2.circle(img, (20, 20), radius, r, thickness)
            cv2.putText(img, "Straight", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, r)
        if self.left:
            cv2.circle(img, (20, 50), radius, g, thickness)
            cv2.putText(img, "Left", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, g)
        else:
            cv2.circle(img, (20, 50), radius, r, thickness)
            cv2.putText(img, "Left", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, r)
        if self.right:
            cv2.circle(img, (20, 80), radius, g, thickness)
            cv2.putText(img, "Right", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, g)
        else:
            cv2.circle(img, (20, 80), radius, r, thickness)
            cv2.putText(img, "Right", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, r)
        if(self.straight and self.left and self.right):
            cv2.rectangle(img, top_left, bottom_right, g, 1)
        else:    
            cv2.rectangle(img, top_left, bottom_right, r, 1)
        cv2.imshow('or', img)

estimator = Head_pose()

while True:
    ret, img = cap.read()
    if ret:
        estimator.estimate_pose(img,300, 400, .1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break













# # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# center_coordinates = (int(img.shape[1]/2), int(img.shape[0]/2))
# print(center_coordinates)
# axesLength = (150, 200)
# top_left = (int(center_coordinates[0]-axesLength[0]), int(center_coordinates[1]-axesLength[1]))
# bottom_right = (int(center_coordinates[0]+axesLength[0]), int(center_coordinates[1]+axesLength[1]))
# cut_img = np.copy(img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :])
# print(top_left)
# print(bottom_right)
# angle = 0  
# startAngle = 0
# endAngle = 360
# # Red color in BGR 
# color = (0, 0, 255)     
# cv2.rectangle(img, top_left, bottom_right, color, 3)
# # Line thickness of 5 px 
# thickness = 5
# cv2.ellipse(img, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness) 
# cv2.imshow('frame', cut_img)



