import cv2
import torch
import torch.nn.functional as F
import numpy as np
from lane_detector import LaneDetector
from utils.utils import (
    time_synchronized,
    select_device,
    scale_coords,
    xyxy2xywh,
    non_max_suppression,
    split_for_trace_model,
    driving_area_mask,
    lane_line_mask,
    letterbox,
)


class YOLOPv2Detector:
    def __init__(
        self,
        weights="weights/yolopv2.pt",
        img_size=640,
        device="0",
        conf_thres=0.3,
        iou_thres=0.45,
    ):
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = select_device(device)
        self.stride = 32

        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Device selected: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
            )

        self.model = torch.jit.load(weights)
        self.model.to(self.device)
        self.half = self.device.type != "cpu"
        if self.half:
            self.model.half()
        self.model.eval()

        if self.device.type != "cpu":
            self.model(
                torch.zeros(1, 3, img_size, img_size)
                .to(self.device)
                .type_as(next(self.model.parameters()))
            )

        self.lane_detector = LaneDetector(self.device)

    def preprocess_frame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img, ratio, (dw, dh) = letterbox(
            img, new_shape=(self.img_size, self.img_size), stride=self.stride
        )
        img = torch.from_numpy(img.transpose(2, 0, 1)).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img, ratio, (dw, dh)

    def detect(self, frame):
        im0 = frame.copy()

        t_pre = time_synchronized()
        img, ratio, pad = self.preprocess_frame(frame)
        t_pre_end = time_synchronized()

        print(f"Model device: {next(self.model.parameters()).device}")
        print(f"Input tensor device: {img.device}")

        t_inf = time_synchronized()
        [pred, anchor_grid], seg, ll = self.model(img)
        t_inf_end = time_synchronized()

        t_post = time_synchronized()
        # Object detection
        pred = split_for_trace_model(pred, anchor_grid)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

        # GPU-based mask and overlay
        da_seg_mask = torch.from_numpy(driving_area_mask(seg)).to(self.device)  # [H, W]
        ll_seg_mask = torch.from_numpy(lane_line_mask(ll)).to(self.device)  # [H, W]
        masks = torch.stack([da_seg_mask, ll_seg_mask]).unsqueeze(0)  # [1, 2, H, W]
        masks = F.interpolate(masks, size=im0.shape[:2], mode="nearest").squeeze(
            0
        )  # [2, H, W]

        # Single overlay tensor
        palette = torch.tensor(
            [[0, 0, 0], [0, 255, 0], [0, 0, 255]],
            dtype=torch.float32,
            device=self.device,
        )  # BGR
        overlay = torch.zeros((im0.shape[0], im0.shape[1], 3), device=self.device)
        overlay[masks[0] == 1] = palette[1]  # Green for drivable area
        overlay[masks[1] == 1] = palette[2]  # Red for lanes

        # Blend with frame
        im0_tensor = torch.from_numpy(im0).to(self.device).float()  # [H, W, 3]
        color_mask = overlay.mean(dim=2) > 0  # [H, W]
        im0_tensor[color_mask] = (
            im0_tensor[color_mask] * 0.5 + overlay[color_mask] * 0.5
        )

        # GPU-based bounding boxes
        if pred is not None:
            for det in pred:
                if det is not None and len(det):
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape, ratio_pad=(ratio, pad)
                    ).round()
                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = map(int, xyxy)
                        # Draw rectangle on GPU tensor
                        im0_tensor[y1:y2, x1 : x1 + 2, :] = torch.tensor(
                            [0, 255, 255], device=self.device
                        )  # Left
                        im0_tensor[y1:y2, x2 - 2 : x2, :] = torch.tensor(
                            [0, 255, 255], device=self.device
                        )  # Right
                        im0_tensor[y1 : y1 + 2, x1:x2, :] = torch.tensor(
                            [0, 255, 255], device=self.device
                        )  # Top
                        im0_tensor[y2 - 2 : y2, x1:x2, :] = torch.tensor(
                            [0, 255, 255], device=self.device
                        )  # Bottom

        # Finalize frame
        im0 = im0_tensor.cpu().numpy().astype(np.uint8)

        t_post_end = time_synchronized()

        print(f"Preprocessing time: {(t_pre_end - t_pre):.3f}s")
        print(f"Inference time: {(t_inf_end - t_inf):.3f}s")
        print(f"Post-processing time: {(t_post_end - t_post):.3f}s")
        print(f"Total frame time: {(t_post_end - t_pre):.3f}s")
        return im0


if __name__ == "__main__":
    import numpy as np

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector = YOLOPv2Detector()
    result = detector.detect(frame)
    cv2.imshow("Test", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
