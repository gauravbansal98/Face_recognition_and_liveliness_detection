import multiprocessing
import cv2
from multiprocessing import Process, Lock, Queue, Value
import time
from liveliness import blink
from recognition import Face_recognition


def cam_loop(images, end):
    cap = cv2.VideoCapture(0)

    while True:
        _ , img = cap.read()
        if img is not None:
            images.put(img)
        if end.value == 1:
            cap.release()
            return

def eye_blink_detection(images, end, lock, live):
    detector = blink()
    initial_time = time.time()
    while True:
        if end.value == 1:
            return
        if(time.time()-initial_time > 5):
            live.value = 0
        else:
            live.value = 1
        if images.empty() == False:
            img = images.get()
            (img, detection) = detector.detect_blink(img)
            if detection == 1:
                initial_time = time.time()
                print("Liveliness")
            # cv2.imshow('eye_blink_detection', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                lock.acquire()
                end.value = 1
                lock.release()
                return

def face_recognition_function(images, end, lock, live):
    recognizer = Face_recognition()
    while True:
        if end.value == 1:
            return
        if images.empty() == False:
            img = images.get()
            img = recognizer.recognize(img)
            if(live.value == 1):
                cv2.putText(img, "Real person", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            else:
                cv2.putText(img, "Person not real", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            cv2.imshow('face_recognition', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                lock.acquire()
                end.value = 1
                lock.release()
                return

if __name__ == '__main__':

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    lock = Lock()
    images = Queue(1)
    end = Value('i', 0)
    live = Value('i', 0)

    cam_process = Process(target=cam_loop,args=(images, end))
    cam_process.start()

    eye_blink = Process(target=eye_blink_detection,args=(images, end, lock, live))
    eye_blink.start()

    face_recog = Process(target=face_recognition_function,args=(images, end, lock, live))
    face_recog.start()


    eye_blink.join()
    face_recog.join()
    eye_blink.terminate()
    face_recog.terminate()
    cam_process.terminate()
    cam_process.join()

cv2.destroyAllWindows()