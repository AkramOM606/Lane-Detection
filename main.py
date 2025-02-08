# Using pyscreenshot (alternative to Telemetry)
# from PIL import ImageGrab
import pyscreenshot as ImageGrab

img = ImageGrab.grab(bbox=(2, 36, 1280, 720 + 36), backend="mss", childprocess=False)
# X1, Y1, X2, Y2 # zero 'childprocess' and 'mss' gives the best performance in most cases
