from ultralytics import YOLO
from IPython.display import display, Image
import os
import sys

# TODO - Fill path to image or dict
video = 'path_to_video.mp4'

# choose model, TS_CV is the final best model
model = YOLO('Model/TS_CV.pt')

# run prediction
results = model.predict(source=video,
                        save=True, stream=True, vid_stride=1, save_txt=True,
                        save_conf=True, save_frames=False, show_labels=True,
                        show_conf=True, show_boxes=True)
for result in results:
    boxes = result.boxes