import cv2
import numpy as np
import glob


def white_balance(img):
    # normalize color in image
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def dilate(img):
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    return dilation


def distance(keypoint1, keypoint2):
    arr = np.array([keypoint1.pt, keypoint2.pt])
    dst = np.sqrt(np.sum((arr[0] - arr[1])**2))
    return dst


def smallest_distance(key1, key_lst2):
    # find the smallest distance of a keypoint and a list of keypoints
    pos = -1
    dst = 999
    for i, key2 in enumerate(key_lst2):
        tmp = distance(key1, key2)
        if tmp < dst:
            dst = tmp
            pos = i
    return dst, pos

def smallest_distance2(key_lst1, key_lst2):
    # find the smallest distance in two list of keypoints
    dst = 999
    for i, key1 in enumerate(key_lst1):
        for j, key2 in enumerate(key_lst2):
            tmp = distance(key1, key2)
            if tmp < dst:
                dst = tmp
                pos_1 = i
                pos_2 = j
    return dst, pos_1, pos_2

def filter_color(frame, color="all"):
    # filter an image frame with a given color of the following:
    # red, blue, green, yellow and white
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if color == "red":
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        lower_red2 = np.array([170, 150, 70])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red, upper_red)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = red_mask1 + red_mask2
        return cv2.bitwise_and(frame, frame, mask=red_mask), red_mask
    elif color == "blue":
        lower = np.array([100, 80, 0])
        upper = np.array([150, 255, 255])
    elif color == "green":
        lower = np.array([40, 120, 40])
        upper = np.array([90, 255, 255])
    elif color == "yellow":
        lower = np.array([10, 180, 160])
        upper = np.array([80, 255, 255])
    elif color == "white":
        lower = np.array([0, 0, 160])
        upper = np.array([255, 40, 255])
    elif color == "black":
        lower = np.array([0, 0, 0])
        upper = np.array([255, 255, 75])
    else:
        return frame, frame
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(frame, frame, mask=mask), mask


def calibrate_offline():
    # Defining the dimensions of checkerboard
    # not finding the chessboard corners right now
    CHECKERBOARD = (7, 5)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    images = glob.glob('./pics/calibration/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        cv2.imshow('img', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    h, w = img.shape[:2]

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)

# TODO still open
def calibrate(self):
    # here should be a calibration function as we should be able to use different webcams and phones
    patternsize = (7, 7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    imgpoints = []
    objp = np.zeros((1, patternsize[0] * patternsize[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:patternsize[0], 0:patternsize[1]].T.reshape(-1, 2)
    prev_img_shape = None
    while True:
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
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

