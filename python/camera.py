import numpy as np
import cv2
from matplotlib import pyplot as plt


class Camera:
    def __init__(self, video_feed=0):
        self.cam = cv2.VideoCapture(video_feed)
        # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def show_video(self):
        while(True):
            ret, frame = self.cam.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()

    def take_picture(self):
        _, img = self.cam.read()
        cv2.imwrite("output_image.png", img)
        return img

    def load_test_image(self):
        img = cv2.imread("pics/chess.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (600, 400))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.show()
        return img

    def find_features(self):
        img = self.load_test_image()
        fast = cv2.FastFeatureDetector()
        kp = fast.detect(img, None)
        img2 = cv2.drawKeypoints(img, kp, color=(0, 255, 0))
        plt.imshow("fast", img2), plt.show()

    def calibrate(self):
        # here should be a calibration function as we should be able to use different webcams and phones
        patternsize = (7, 7)
        # _, frame = self.cam.read()
        frame = self.load_test_image()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        patternfound, corners = cv2.findChessboardCorners(gray, patternsize)
        if patternfound:
            print("pattern found @: ", corners)
            cv2.cornerSubPix()
            plt.imshow(gray), plt.show()
        else:
            print("no pattern found")

        # cv2.calibrateCamera()

