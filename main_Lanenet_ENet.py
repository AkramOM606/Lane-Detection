import time
import cv2

from game_capture import capture_game_window, get_game_window

# from lane_detection import detect_lanes, draw_lines, preprocess_image
from lanenet.lane_detection import detect_lanes
from object_detection import detect_objects

game_window = get_game_window()

while True:
    start_time = time.time()
    frame = capture_game_window(game_window)

    if frame is not None:
        # Lane detection here !!!
        # edges = preprocess_image(frame)
        # lines = detect_lanes(edges)
        # annotated_frame = draw_lines(frame, lines)
        # final_frame = annotated_frame

        lane_frame = detect_lanes(frame)
        # final_frame = detect_objects(lane_frame)

        cv2.imshow("Combined Detection", lane_frame)

    if cv2.waitKey(1) == ord("q"):
        break

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"FPS: {fps:.2f}")

cv2.destroyAllWindows()
