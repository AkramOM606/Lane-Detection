import pygetwindow as gw
import win32gui
from mss import mss
import numpy as np
import cv2


def get_game_window():
    windows = gw.getAllTitles()
    for title in windows:
        if "Euro Truck Simulator 2" in title:
            game_window = gw.getWindowsWithTitle(title)[0]
            return game_window
    raise Exception("Euro Truck Simulator 2 window not found")


def capture_game_window(game_window):
    with mss() as sct:
        hwnd = game_window._hWnd
        # Get client area
        rect = win32gui.GetClientRect(hwnd)
        client_left, client_top, client_right, client_bottom = rect
        width = client_right - client_left
        height = client_bottom - client_top

        # Convert to screen coordinates
        client_top_left = win32gui.ClientToScreen(hwnd, (client_left, client_top))
        screen_left = client_top_left[0]
        screen_top = client_top_left[1]

        monitor = {
            "top": screen_top,
            "left": screen_left,
            "width": width,
            "height": height,
        }

        # Debug: Print the capture region
        print(
            f"Capture region: top={monitor['top']}, left={monitor['left']}, "
            f"width={monitor['width']}, height={monitor['height']}"
        )

        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Debug: Print frame shape
        print(f"Captured frame shape: {frame.shape}")

        return frame
