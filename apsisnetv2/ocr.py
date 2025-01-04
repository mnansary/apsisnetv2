#--------------------------------
# imports
#--------------------------------

import os
import cv2
import time
import math
import numpy as np
import gdown
import copy
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Union,List,Dict

from .robustscanner import RobustScanner
from apsisocr.paddledbnet import PaddleDBNet
from apsisocr.utils import download,LOG_INFO

from ultralytics import YOLO
from pycocotools import mask as cocomask

from .rotation import auto_correct_image_orientation
from .serialization import assign_line_word_numbers
from .merging import merge_segments

#-------------------------------------------
# using gpu
# #--------------------------------------------
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

class ImageOCR(object):
    def __init__(self, 
                 robust_scanner_weights_dir:str="model/robust_scanner_resnet50") -> None:
        """
        Initializes the ImageOCR class by loading the YOLO and OCR models.

        Args:
            yolo_dla_model_path (str): Path to the YOLO model weights.
            yolo_dla_model_gid (str): Google Drive ID to download the YOLO model if not found locally.
            yolo_conf_thresh (float) : The confidence threshold for yolo inference
        """
        self.bn_rec=RobustScanner(robust_scanner_weights_dir)
        LOG_INFO("Loaded Bangla Recognition Model: RobustScanner")
        self.detector=PaddleDBNet(load_line_model=False)        
        LOG_INFO("Loaded Word detector Model: Paddle-DBnet")
    
    def process_ocr(self,image:np.ndarray)->Tuple[List[dict],dict]:
        """
            Args:
                image(np.ndarray) : Input Image
            Returns:
                Tuple[List[dict],dict] 
                    - List[dict]: words   - OCR recognition results as words.
                                          - keys for each word ['poly','text',"line_num","word_num"]
                    - dict : rotation_data- orientation information for ocr
                                          - keys for meta_data ["rotated_image","rotated_mask","angle"]   
        """
        # extract word regions
        regions=self.detector.get_word_boxes(image)
        # correct for rotation
        rotated_image,rotated_mask,angle,rotated_polys=auto_correct_image_orientation(image,regions)
        # get word crops
        crops=self.detector.get_crops(rotated_image,rotated_polys)
        # get text
        texts=self.bn_rec.infer(crops)
        # get word and line number
        words=[{"poly":poly,"text":txt} for poly,txt in zip(rotated_polys,texts)]
        words=assign_line_word_numbers(words)
        # format output data
        rotation_data={ "rotated_image":rotated_image,
                        "rotated_mask":rotated_mask,
                        "angle":angle}
        
        return words,rotation_data


    def __call__(self, image: Union[str, np.ndarray]) -> dict:
        """
        Processes an image with YOLO for object detection and OCR for text recognition.

        Args:
            image (Union[str, np.ndarray]): Input image. Can be a file path or a NumPy array.

        Returns:
            dict : that holds two keys: ["words","rotation"]
                    words   - OCR recognition results as words.
                            - keys for each word ['poly','text','line_num','word_num']
                    rotation- orientation information for ocr
                            - keys for meta_data ["rotated_image","rotated_mask","angle"]   
                    
        """
        # If the image is a file path, read and convert it.
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        # Perform OCR on the image
        words,rotation_data= self.process_ocr(image)
        return {"words":words,"rotation":rotation_data}

