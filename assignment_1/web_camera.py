import cv2
import os



cam = cv2.VideoCapture(0)

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
cam_fps = cam.get(cv2.CAP_PROP_FPS)

output_dr = os.path.expanduser('solutions')
os.makedirs(output_dr, exist_ok=True)

output_file = os.path.join(output_dr, 'camera_outputs.txt')

with open(output_file, 'w') as f:
    f.write(f"Frame width: {frame_width} \nFrame height: {frame_height} \nFrame FPS: {cam_fps} \n")

cam.release()
print("Camera information saved to ", output_file)