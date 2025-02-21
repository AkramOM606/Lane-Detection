import cv2
import numpy as np
from utils.utils import lane_line_mask, show_seg_result


class LaneDetector:
    def __init__(self, device):
        self.device = device

    def overlay_lanes(self, frame, ll_seg_mask):
        """Overlay pre-resized lane mask on the frame."""
        # print(f"Lane detection - Frame shape: {frame.shape}")
        # print(f"LL mask shape: {ll_seg_mask.shape}")
        show_seg_result(frame, (np.zeros_like(ll_seg_mask), ll_seg_mask), is_demo=True)


if __name__ == "__main__":
    import torch

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    ll_seg_mask = np.zeros((720, 1280), dtype=np.uint8)  # Mock pre-resized mask
    device = torch.device("cpu")

    lane_detector = LaneDetector(device)
    lane_detector.overlay_lanes(frame, ll_seg_mask)
    cv2.imshow("Lane Test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
