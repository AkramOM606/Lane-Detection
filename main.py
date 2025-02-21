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
        start_time = time.time()
        frame = capture_game_window(game_window)

        if frame is not None:
            # print(f"Captured frame shape: {frame.shape}")
            final_frame = detector.detect(frame)
            cv2.imshow("Combined Detection", final_frame)

        if cv2.waitKey(1) == ord("q"):
            break

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"FPS: {fps:.2f}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    with torch.no_grad():
        main()
