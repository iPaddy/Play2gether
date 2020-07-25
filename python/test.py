import camera

cam = camera.Camera(0)
#cam.find_board(detector="akaze", test=True)
#cam.compare_features("orb", "sift", "kaze", "akaze", "pics/test_board2.jpg")
#cam.find_board(detector="akaze")
#cam.calibrate()
#cam.show_video(color="all")
cam.find_piece()
