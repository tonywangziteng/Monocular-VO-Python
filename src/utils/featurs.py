from asyncio import AbstractServer
import logging
from typing import Any, Tuple
from abc import ABC, abstractmethod

import cv2
import numpy as np

import Detectors
import Descriptors

# TODO: base feature class
class feature(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def detect(
        self, img:np.ndarray, 
        x_cell_num:int=1, y_cell_num:int=1, 
        cell_keypts_max_num:int=None, 
        **kargs
    )->Tuple[cv2.KeyPoint]:
        pass

    @abstractmethod
    def describe(
        self, img:np.ndarray, 
        keypoints:Tuple[cv2.KeyPoint]
    ) -> Tuple[np.ndarray]:
        pass

    @abstractmethod
    def detectAndDescribe(
        self, img:np.ndarray, 
        x_cell_num:int=1, y_cell_num:int=1, 
        cell_keypts_max_num:int=None, 
        **kargs
    )->Tuple[Tuple[cv2.KeyPoint], Tuple[np.ndarray]]:
        pass

# TODO: SIFT feature class
class SIFT(feature):
    def __init__(self) -> None:
        super().__init__()
        self._detector = Detectors.SIFT_Detector(
            contrastThreshold=0.09, 
        )
        self._desciptor = Descriptors.SIFT_descriptor()

    def detect(self, img: np.ndarray, x_cell_num: int = 1, y_cell_num: int = 1, cell_keypts_max_num: int = None, **kargs) -> Tuple[cv2.KeyPoint]:
        super().detect(img, x_cell_num, y_cell_num, cell_keypts_max_num, **kargs)
        return self._detector.detect(img, x_cell_num, y_cell_num, cell_keypts_max_num, **kargs)
        
    def describe(self, img: np.ndarray, keypoints: Tuple[cv2.KeyPoint]) -> Tuple[np.ndarray]:
        super().describe(img, keypoints)
        return self._desciptor.describe(img, keypoints)

    def detectAndDescribe(self, img: np.ndarray, x_cell_num: int = 1, y_cell_num: int = 1, cell_keypts_max_num: int = None, **kargs) -> Tuple[Tuple[cv2.KeyPoint], Tuple[np.ndarray]]:
        super().detectAndDescribe(img, x_cell_num, y_cell_num, cell_keypts_max_num, **kargs)
        keypoints = self._detector.detect(
            img, x_cell_num, y_cell_num, cell_keypts_max_num
        )
        descriptors = self._desciptor.describe(img, keypoints)
        return (keypoints, descriptors)
# TODO: test SIFT class


# TODO: self-defined feature class