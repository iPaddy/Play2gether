import numpy as np
import cv2
from matplotlib import pyplot as plt
import helper


class Camera:
    def __init__(self, video_feed=0):
        self.cam = cv2.VideoCapture(video_feed)
        #self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        #self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        #print(self.cam.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        self.hom = None
        self.frame = self.cam.read()
        self.board_ref = cv2.resize(cv2.imread("pics/board2.jpg"), (1080, 1080))

    def show_video(self, color="all"):
        while True:
            ret, self.frame = self.cam.read()

            if color != "all":
                self.frame, _ = helper.filter_color(self.frame, color)

            if self.hom is not None:
                h_b, w_b, c_w = self.board_ref.shape
                warped = cv2.warpPerspective(self.frame, self.hom, (w_b, h_b))
                # here it should warp correctly but it cuts the lower third
                #warped, _ = helper.filter_color(warped, "white")
                self.find_color_piece(color)
                #cv2.imshow('frame', warped)
            else:
                tmp_frame = self.frame
                self.frame = helper.white_balance(self.frame)
                cv2.imshow('live video', self.frame)
                #self.find_white()
                self.find_color_piece(color)
                #cv2.imshow('frame2', tmp_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        self.cam.release()
        cv2.destroyAllWindows()


    def take_picture(self):
        _, img = self.cam.read()
        cv2.imwrite("pics/test_board.png", img)
        return img

    def load_test_image(self):
        img = cv2.imread("pics/test_board3.jpg")
        #img = cv2.imread("pics/test_board3.jpg")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.show()
        return img

    def load_board_reference(self, detector="akaze"):
        kps, desc = self.find_features(detector, test=False, visual=False, img=self.board_ref)
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.show()
        return kps, desc, self.board_ref

    def find_board(self, min_n_matches=10, test=False, detector="akaze"):
        # board_img = cv2.resize(cv2.imread("pics/board2.jpg"), (640, 480))
        if test:
            img = self.load_test_image()
        else:
            _, img = self.cam.read()
        kp1, desc1, board_img = self.load_board_reference(detector)
        kp2, desc2 = self.find_features(detector, visual=False, img=img)

        flann_index = 1
        index_params = dict(algorithm=flann_index, trees=5)
        search_params = dict(checks=50)

        #flann = cv2.FlannBasedMatcher(index_params, search_params)
        #matches = flann.knnMatch(desc1, desc2, k=2)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING2)
        matches = matcher.knnMatch(desc1, desc2, k=2)
        # store good matches
        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        # enough matches are found
        if len(good) > min_n_matches:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()
            h, w, d = board_img.shape
            # corner points
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # destination of corner points
            dst = cv2.perspectiveTransform(pts, M)
            # draw rectangle around found board
            img2 = cv2.polylines(img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            # transform live view to one viewing from the top of the board
            self.hom, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        else:
            print("Not enough matches are found - {}/{}".format(len(good), min_n_matches))
            matches_mask = None
            return

        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matches_mask, flags=2)
        out_img = cv2.drawMatches(board_img, kp1, img2, kp2, good, None, **draw_params)
        plt.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), 'gray'), plt.show()

    def find_features(self, detector="orb", test=False, visual=False, img_loc=None, img=None):
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
        if not test and img_loc is None and img is None:
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

        # needs opencv-contrib-python and is a pain to set up correctly
        if detector == "sift":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            kps, desc = sift.detectAndCompute(img, None)
            img = cv2.drawKeypoints(img, kps, None, color=(0, 255, 0), flags=0)
            if visual:
                return img
            else:
                return kps, desc

        # needs opencv-contrib-python and is a pain to set up correctly
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

    def compare_features(self, feat1="orb", feat2="surf", feat3="sift", feat4="akaze", img_loc=None):
        """
        compares up to four feature finding functions

        plotting them against each other in a grid
        """
        img0 = self.find_features(feat1, visual=True, img_loc=img_loc)
        img1 = self.find_features(feat2, visual=True, img_loc=img_loc)
        img2 = self.find_features(feat3, visual=True, img_loc=img_loc)
        img3 = self.find_features(feat4, visual=True, img_loc=img_loc)

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
        patternsize = (8, 8)
        _ , frame = self.cam.read()
        #frame = self.load_test_image()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        patternfound, corners = cv2.findChessboardCorners(gray, patternsize)
        if patternfound:
            print("pattern found @: ", corners)
            cv2.cornerSubPix()
            plt.imshow(gray), plt.show()
        else:
            print("no pattern found")

        # cv2.calibrateCamera()

    def find_white(self):
        # used to find the white on top of the pieces
        _, img = self.cam.read()
        img, mask = helper.filter_color(img, "white")
        mask = cv2.bitwise_not(mask)
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 1
        params.maxThreshold = 255
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 200
        params.filterByCircularity = True
        params.minCircularity = 0.4
        params.maxCircularity = 1
        params.filterByConvexity = True
        params.minConvexity = 0.2
        params.maxConvexity = 1
        params.filterByInertia = True
        params.minInertiaRatio = 0.3
        params.maxInertiaRatio = 1
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(mask)
        img_with_detections = cv2.drawKeypoints(mask, keypoints, np.array([]), (0, 0, 255),
                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Detected Blobs", img_with_detections)
        #cv2.waitKey(0)

    def find_color_piece(self, color="red"):
        # find bigger blobs of the play pieces
        _, img = self.cam.read()
        img, mask = helper.filter_color(img, color)
        mask = cv2.bitwise_not(mask)
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 1
        params.maxThreshold = 255
        params.filterByArea = True
        params.minArea = 300
        params.maxArea = 3000
        params.filterByCircularity = True
        params.minCircularity = 0.2
        params.maxCircularity = 1
        params.filterByConvexity = True
        params.minConvexity = 0
        params.maxConvexity = 1
        params.filterByInertia = True
        params.minInertiaRatio = 0.2
        params.maxInertiaRatio = 1
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(mask)
        img_with_detections = cv2.drawKeypoints(mask, keypoints, np.array([]), (0, 0, 255),
                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow(color + " Blobs", img_with_detections)
        #cv2.waitKey(0)

    def board_positions(self):
        self.load_board_reference()
