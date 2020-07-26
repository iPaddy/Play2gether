import cv2
import numpy as np


def white_balance(img):
    # normalize color in image
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


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
    if color == "blue":
        lower_blue = np.array([100, 80, 0])
        upper_blue = np.array([150, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        return cv2.bitwise_and(frame, frame, mask=blue_mask), blue_mask
    if color == "green":
        lower_green = np.array([40, 120, 40])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        return cv2.bitwise_and(frame, frame, mask=green_mask), green_mask
    if color == "yellow":
        lower_yellow = np.array([18, 180, 180])
        upper_yellow = np.array([80, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        return cv2.bitwise_and(frame, frame, mask=yellow_mask), yellow_mask
    if color == "white":
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([255, 40, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        return cv2.bitwise_and(frame, frame, mask=white_mask), white_mask
