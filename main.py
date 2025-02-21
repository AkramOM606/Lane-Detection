import time
import cv2
import torch
from game_capture import capture_game_window, get_game_window
from yolopv2_detector import YOLOPv2Detector  # Import the new detector


def main():
    game_window = get_game_window()
    detector = YOLOPv2Detector(
        weights="weights/yolopv2.pt",
        img_size=640,
        device="0",
        conf_thres=0.3,
        iou_thres=0.45,
    )
    while True:
        t_start = time.time()
        frame = capture_game_window(game_window)
        t_capture = time.time()
        if frame is not None:
            final_frame = detector.detect(frame)
            cv2.imshow("Combined Detection", final_frame)
        t_end = time.time()
        print(f"Capture time: {(t_capture - t_start):.3f}s")
        print(f"Total time: {(t_end - t_start):.3f}s")
        print(f"FPS: {1 / (t_end - t_start):.2f}")
        if cv2.waitKey(1) == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    with torch.no_grad():
        main()
