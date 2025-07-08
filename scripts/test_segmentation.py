import os
import cv2
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from skimage import exposure
from collections import deque

from TumorSegmentationApp import TumorSegmentationApp
from metrics import dice_loss, dice_coef, iou_metric
from models import TumorSegmentationModel
from processors import ImageProcessor



if __name__ == "__main__":
    root = tk.Tk()
    model_path = os.path.abspath("unet_model.h5")

    if not os.path.exists(model_path):
        new_model = TumorSegmentationModel().model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        new_model.save(model_path)

    app = TumorSegmentationApp(root, model_path=model_path)
    root.mainloop()