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
        img = cv2.imread("pics/chess2.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (600, 400))
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.show()
        return img

    def find_features(self):
        # load test image
        img = self.load_test_image()

        # fast feature detector
        fast = cv2.FastFeatureDetector.create()
        kpf = fast.detect(img, None)
        img_fast = cv2.drawKeypoints(img, kpf, None, color=(0, 255, 0), flags=0)

        # orb feature detector
        orb = cv2.ORB_create()
        kpo = orb.detect(img, None)
        kpo, des = orb.compute(img, kpo)
        img_orb = cv2.drawKeypoints(img, kpo, None, color=(0, 255, 0), flags=0)

        # shi-tomasi corner detector
        img_shi = img
        corners = cv2.goodFeaturesToTrack(img_shi, maxCorners=100, qualityLevel=0.01, minDistance=10)
        corners - np.int0(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img_shi, center=(x, y), radius=5, color=(0, 255, 0), thickness=-1)

        # harris corner detector
        thresh = 150
        img_harris = img
        img_harris = np.float32(img_harris)
        dst = cv2.cornerHarris(img_harris, blockSize=2, ksize=3, k=0.04)
        dst_norm = np.empty(dst.shape, dtype=np.float32)
        cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
        for i in range(dst_norm.shape[0]):
            for j in range(dst_norm.shape[1]):
                if int(dst_norm[i, j]) > thresh:
                    cv2.circle(dst_norm_scaled, center=(j, i), radius=5, color=0)

        fig = plt.figure(figsize=(4, 4))
        fig.add_subplot(2, 2, 1)
        plt.imshow(img_orb)
        fig.add_subplot(2, 2, 2)
        plt.imshow(img_fast)
        fig.add_subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(img_shi, cv2.COLOR_BGR2RGB))
        fig.add_subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(dst_norm_scaled, cv2.COLOR_BGR2RGB))

        plt.show()

    def find_features_fast(self):
        img = self.load_test_image()
        fast = cv2.FastFeatureDetector()
        kp = fast.detect(img, None)
        kp, des = fast.compute(img, kp)
        img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        plt.imshow(img2), plt.show()

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

