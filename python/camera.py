import numpy as np
import cv2
from matplotlib import pyplot as plt


class Camera:
    def __init__(self, video_feed=0):
        self.cam = cv2.VideoCapture(video_feed)

    def show_video(self):
        while (True):
            ret, frame = self.cam.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()

    def take_picture(self):
        _, img = self.cam.read()
        cv2.imwrite("pics/image_capture.png", img)
        return img

    def load_test_image(self):
        img = cv2.imread("pics/chess2.jpg")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.show()
        return img

    def load_board_reference(self):
        img = cv2.resize(cv2.imread("pics/board.jpg"), (640, 480))
        kps, desc = self.find_features("surf", test=False, visual=False, img=img)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.show()
        return kps, desc

    def find_features(self, detector="orb", test=False, visual=True, img_loc=None, img=None):
        """
        Function for choosing different feature detectors

        :param detector:string, choose a detector
        :param test:bool, if True it reads from a test image, otherwise from a video stream
        :param visual:bool decides if an image is returned or the descriptors etc
        depends on the detection used how many and which arguments are returned. Take care
        :param img_loc:string, path to a image to be analysed
        :param img:np.array
        :return: image or descriptors
        """

        if test and img_loc is None and img is None:
            # load test image
            img = self.load_test_image()
        if img_loc is None and img is None:
            _, img = self.cam.read()
        try:
            if img_loc is not None:
                img = cv2.imread(img_loc)
        except:
            print("could not read image")

        # fast feature detector
        if detector == "fast":
            fast = cv2.FastFeatureDetector.create()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kpf = fast.detect(img, None)
            img = cv2.drawKeypoints(img, kpf, None, color=(0, 255, 0), flags=0)
            if visual:
                return img
            else:
                return kpf

        # orb feature detector
        if detector == "orb":
            orb = cv2.ORB_create()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kpo = orb.detect(img, None)
            kpo, des = orb.compute(img, kpo)
            img = cv2.drawKeypoints(img, kpo, None, color=(0, 255, 0), flags=0)
            if visual:
                return img
            else:
                return kpo, des

        # shi-tomasi corner detector
        if detector == "shi":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(img, maxCorners=100, qualityLevel=0.01, minDistance=10)
            corners - np.int0(corners)
            for i in corners:
                x, y = i.ravel()
                cv2.circle(img, center=(x, y), radius=3, color=(255, 0, 255), thickness=-1)
            if visual:
                return img
            else:
                return corners

        # harris corner detector
        if detector == "harris":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = 150
            img = np.float32(img)
            dst = cv2.cornerHarris(img, blockSize=2, ksize=3, k=0.04)
            dst_norm = np.empty(dst.shape, dtype=np.float32)
            cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
            for i in range(dst_norm.shape[0]):
                for j in range(dst_norm.shape[1]):
                    if int(dst_norm[i, j]) > thresh:
                        cv2.circle(dst_norm_scaled, center=(j, i), radius=5, color=0)
            if visual:
                return dst_norm_scaled
            else:
                return dst, img

        if detector == "sift":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            kps, desc = sift.detectAndCompute(img, None)
            img = cv2.drawKeypoints(img, kps, None, color=(0, 255, 0), flags=0)
            if visual:
                return img
            else:
                return kps, desc

        if detector == "surf":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            surf = cv2.xfeatures2d.SURF_create()
            kps, desc = surf.detectAndCompute(img, None)
            img = cv2.drawKeypoints(img, kps, None, color=(0, 255, 0), flags=0)
            if visual:
                return img
            else:
                return kps, desc

        if detector == "kaze":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kaze = cv2.KAZE_create()
            kps, desc = kaze.detectAndCompute(img, None)
            img = cv2.drawKeypoints(img, kps, None, color=(0, 255, 0), flags=0)
            if visual:
                return img
            else:
                return kps, desc

        if detector == "akaze":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            akaze = cv2.AKAZE_create()
            kps, desc = akaze.detectAndCompute(img, None)
            img = cv2.drawKeypoints(img, kps, None, color=(0, 255, 0), flags=0)
            if visual:
                return img
            else:
                return kps, desc

        if detector == "brisk":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brisk = cv2.BRISK_create()
            kps, desc = brisk.detectAndCompute(img, None)
            img = cv2.drawKeypoints(img, kps, None, color=(0, 255, 0), flags=0)
            if visual:
                return img
            else:
                return kps, desc

        else:
            print("detector not implemented. Try using one of the following: orb, fast, shi, harris")

    def compare_features(self, feat1="orb", feat2="surf", feat3="sift", feat4="akaze"):
        """
        compares up to four feature finding functions

        plotting them against each other in a grid
        """
        img0 = self.find_features(feat1)
        img1 = self.find_features(feat2)
        img2 = self.find_features(feat3)
        img3 = self.find_features(feat4)

        fig = plt.figure(figsize=(4, 4))
        fig.add_subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
        fig.add_subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        fig.add_subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        fig.add_subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))

        plt.show()

    def find_features_video(self, detector="shi"):

        while (True):
            # read from video input
            _, img = self.cam.read()

            img = self.find_features(detector, test=False)

            cv2.imshow('frame', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()

    # TODO still open
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
