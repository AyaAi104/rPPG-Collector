import cv2
from nexigo_camera import Camera
from config import data_settings as settings
#0 for NEXIGO

#1 for iPhone
camera = Camera(settings["camera_index"])

camera.preview()
