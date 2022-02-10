import logging
from typing import Any, Tuple
from abc import ABC, abstractmethod

import cv2
import numpy as np

from src.utils.Detectors import SIFT_Detector

class Descriptor(ABC):
    def __init__(self, **kargs) -> None:
        super().__init__()

    def describe(
        self, img:np.ndarray, 
        keypoints:Tuple[cv2.KeyPoint]
    ) -> Tuple[np.ndarray]:
        """
        Computing the descriptor the given keypoint. 

        :params img: ndarray, gray scale image
        :params keypoints: Tuple[keyPoint], tuple of input keypoint 

        :return: Tuple[ndarray], tuple of descriptors
        """
        _img = img
        if len(_img.shape) == 3:
            _img = cv2.cvtColor(_img)

        descriptors = self._compute(img, keypoints)

        return descriptors

    @abstractmethod
    def _compute(
        self, 
        img:np.ndarray, 
        keypoints:Tuple[cv2.KeyPoint], 
        **kargs
    ) -> np.ndarray:
        pass


class SIFT_descriptor(Descriptor):
    def __init__(self) -> None:
        super(SIFT_Detector, self).__init__()
    

    def _compute(
        self, 
        img:np.ndarray, 
        keypoints:Tuple[cv2.KeyPoint]
    ) -> np.ndarray:
        # the return is (keypoints, descriptor)
        descriptors = self._sift_detector.compute(img, keypoints, None)

        return descriptors[1]