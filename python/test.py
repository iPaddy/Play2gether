import camera
import helper

cam = camera.Camera(0)
cam.find_board(detector="akaze", show_results=False)
cam.show_video(color="all")


### Testing
#cam.board_positions(visual=True)
#cam.compare_features("orb", "sift", "kaze", "akaze", "pics/test_board2.jpg")
#cam.show_video(color="yellow")
#cam.find_board(detector="akaze", test=True)
#while True:
    #cam.find_white(True)
#cam.find_color_piece()

### Single use
#helper.calibrate_offline()
