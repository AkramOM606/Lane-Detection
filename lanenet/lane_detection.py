import os
import sys

# print("Current working directory:", os.getcwd())
# os.chdir("lanenet")  # Move into the correct directory
# print("Changed working directory to:", os.getcwd())
# print("Current working directory:", os.getcwd())

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Import the LaneNet model from the repository.
# Adjust the import path if needed.
from lanenet.model.lanenet.LaneNet import LaneNet

# Define some constants.
RESIZE_HEIGHT = 256  # Height expected by the model
RESIZE_WIDTH = 512  # Width expected by the model
MODEL_PATH = "./lanenet/log/best_model.pth"  # Update with your weights file path

# Set device: GPU if available, else CPU.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the transform used during testing.
data_transform = transforms.Compose(
    [
        transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Initialize and load the LaneNet model.
# Note: The test.py uses an argument "arch" (model type); adjust if your model requires it.
model = LaneNet(arch="ENet")
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()


def detect_lanes(frame):
    """
    Given an input frame (as captured by OpenCV in BGR format), this function:
      1. Converts and resizes the frame to the model's expected input size.
      2. Applies the necessary normalization and tensor conversion.
      3. Feeds the tensor to the LaneNet model.
      4. Processes the model output to obtain a binary segmentation mask.
      5. Creates a colored overlay (green) where lane pixels are detected.
      6. Blends the overlay with the original frame and returns the result.
    """
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame).resize(
        (RESIZE_WIDTH, RESIZE_HEIGHT), Image.BILINEAR
    )

    transformed_frame = data_transform(pil_img)

    dummy_input = transformed_frame.to(DEVICE)
    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    outputs = model(dummy_input)

    input = Image.open(img_path)
    input = input.resize((512, 256))
    input = np.array(input)

    # instance_pred = (
    #     torch.squeeze(outputs["instance_seg_logits"].detach().to("cpu")).numpy() * 255
    # )
    binary_pred = torch.squeeze(outputs["binary_seg_pred"]).to("cpu").numpy() * 255

    # cv2.imwrite("input.jpg", input)
    # cv2.imwrite(
    #     "instance_output.jpg",
    #     instance_pred.transpose((1, 2, 0)),
    # )
    # cv2.imwrite("binary_output.jpg", binary_pred)

    binary_pred = binary_pred.astype(np.uint8)

    binary_pred = cv2.resize(
        binary_pred, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    return binary_pred

    # # Convert BGR (OpenCV) image to RGB and then to a PIL Image.
    # f()
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # pil_img = Image.fromarray(rgb_frame).resize(
    #     (RESIZE_WIDTH, RESIZE_HEIGHT), Image.BILINEAR
    # )

    # # Apply transformation.
    # input_tensor = data_transform(pil_img).unsqueeze(0).to(DEVICE)

    # # Run inference.
    # with torch.no_grad():
    #     outputs = model(input_tensor)

    # # Process the outputs.
    # # The model returns a dict; we use the binary segmentation prediction.
    # binary_pred = torch.squeeze(outputs["binary_seg_pred"]).to("cpu").numpy() * 255
    # binary_pred = binary_pred.astype(np.uint8)

    # # Resize the binary mask back to the original frame dimensions.
    # mask = cv2.resize(
    #     binary_pred, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
    # )

    # # Create an overlay image: lane regions will be green.
    # overlay = np.zeros_like(frame, dtype=np.uint8)
    # overlay[mask > 128] = [0, 255, 0]

    # # Blend the overlay with the original frame.
    # blended = cv2.addWeighted(frame, 1.0, overlay, 0.5, 0)
    return blended


img_path = "./euro.jpg"


# For standalone testing, you can run this module directly.
def f():
    data_transform = transforms.Compose(
        [
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(img_path)
    img = data_transform(img)

    dummy_input = img.to(DEVICE)
    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    outputs = model(dummy_input)

    input = Image.open(img_path)
    input = input.resize((512, 256))
    input = np.array(input)

    instance_pred = (
        torch.squeeze(outputs["instance_seg_logits"].detach().to("cpu")).numpy() * 255
    )
    binary_pred = torch.squeeze(outputs["binary_seg_pred"]).to("cpu").numpy() * 255

    cv2.imwrite("input.jpg", input)
    cv2.imwrite(
        "instance_output.jpg",
        instance_pred.transpose((1, 2, 0)),
    )
    cv2.imwrite("binary_output.jpg", binary_pred)
