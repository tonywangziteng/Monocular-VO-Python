from typing import Any
import cv2
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Any
from enum import Enum

from src.utils.Detectors import Harris


class VO_MONO(ABC):
    def __init__(
        self, K:np.ndarray, 
        feat_detector_type:str = "SIFT", 
        feat_descriptor_type:str = "SIFT"
    ) -> bool:
        super().__init__()
        self._K = K
        self._feat_detector = None
        self._feat_descriptor = None