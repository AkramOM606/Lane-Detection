import pygetwindow as gw
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


game_window = get_game_window()
print(f"Game window found: {game_window.title}")


def capture_game_window(game_window):
    with mss() as sct:
        monitor = {
            "top": game_window.top,
            "left": game_window.left,
            "width": game_window.width,
            "height": game_window.height,
        }
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR
        return frame
