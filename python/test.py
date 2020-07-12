import camera

cam = camera.Camera()
cam.find_board(detector="akaze", test=True)
#cam.compare_features("orb", "sift", "kaze", "akaze", "pics/test_board2.jpg")
