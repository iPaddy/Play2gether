import camera


def main():
    cam = camera.Camera()
    # cam.take_picture()
    # cam.calibrate()
    # cam.show_video()
    # cam.compare_features()
    cam.find_features_video(detector="shi")


if __name__ == '__main__':
    main()
