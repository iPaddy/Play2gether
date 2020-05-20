import numpy as np
import cv2


def test():
    cam = cv2.VideoCapture(0)
    #cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while(True):
        ret, frame = cam.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def calibrate():
    # here should be a calibration function as we should be able to use different webcams and phones
    print("not yet implemented")
