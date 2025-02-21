import cv2
import torch
import torch.nn.functional as F  # Add this for GPU interpolation
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
    lane_line_mask,  # Add this for direct mask access
    plot_one_box,
    show_seg_result,
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
        # print(f"Preprocessed shape: {img.shape}, ratio: {ratio}, padding: ({dw}, {dh})")

        img = torch.from_numpy(img.transpose(2, 0, 1)).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img, ratio, (dw, dh)

    def detect(self, frame):
        im0 = frame.copy()
        # print(f"Original frame shape: {im0.shape}")

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

        # GPU-based mask processing
        da_seg_mask = torch.from_numpy(driving_area_mask(seg)).to(
            self.device
        )  # Convert to tensor on GPU
        ll_seg_mask = torch.from_numpy(lane_line_mask(ll)).to(
            self.device
        )  # Direct mask access
        masks = torch.stack([da_seg_mask, ll_seg_mask]).unsqueeze(
            0
        )  # Stack for batch resize
        masks = F.interpolate(masks, size=im0.shape[:2], mode="nearest").squeeze(
            0
        )  # Resize on GPU
        da_seg_mask, ll_seg_mask = (
            masks[0].cpu().numpy(),
            masks[1].cpu().numpy(),
        )  # Back to CPU for overlay

        # Bounding boxes
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape, ratio_pad=(ratio, pad)
                ).round()
                for *xyxy, conf, cls in reversed(det):
                    plot_one_box(xyxy, im0, line_thickness=2)

        # Overlays
        self.lane_detector.overlay_lanes(im0, ll_seg_mask)  # Updated method
        show_seg_result(im0, (da_seg_mask, np.zeros_like(da_seg_mask)), is_demo=True)

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
