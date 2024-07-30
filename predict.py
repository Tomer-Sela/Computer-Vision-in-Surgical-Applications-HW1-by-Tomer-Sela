from ultralytics import YOLO
from IPython.display import display, Image
import os
import sys

# TODO - Fill path to image or dict
image = 'path_to_image'

# my model, TS_CV is the final best model
model_name = 'Model/TS_CV.pt'

# run prediction
results = model.predict(source=image,
                        save=True, stream=True, vid_stride=1, save_txt=True,
                        save_conf=True, save_frames=True, show_labels=True,
                        show_conf=True, show_boxes=True)
for result in results:
    result.show()  # display to screen