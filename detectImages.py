# import packages
from IPython.display import Image  # for displaying images
import os 
import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image


# specify directories
input_dir = 'data/images'
output_dir = 'data/results'

# load model
print("Loading model...\n")
model = YOLO('best.pt')

# predict
print("Detecting weed/crops...\n")
results = model.predict(source=input_dir)

# save output results
for i, result in enumerate(results):
    pf = os.path.join(output_dir, str(i) + '.png')
    print("Saving processed image: {}".format(pf))
    res_plotted = result.plot()
    cv2.imwrite(pf, res_plotted)
    