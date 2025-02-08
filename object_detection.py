from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # Load a pretrained model


def detect_objects(frame):
    results = model(frame)
    annotated_frame = results[0].plot()
    return annotated_frame
