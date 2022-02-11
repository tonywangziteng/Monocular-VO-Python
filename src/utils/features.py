from typing import Any, Tuple
from abc import abstractmethod

import cv2
import numpy as np

import utils.Detectors as Detectors
import utils.Descriptors as Descriptors

from utils.utils import baseClass

class feature(baseClass):
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

class SIFT(feature):
    def __init__(self, **kargs) -> None:
        """
        :param nOctaveLayers: int=3, 
            The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.\n
        :param contrastThreshold: float=0.04, 
            The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
            note The contrast threshold will be divided by nOctaveLayers when the filtering is applied. When nOctaveLayers is set to default and if you want to use the value used in D. Lowe paper, 0.03, set this argument to 0.09.\n
        :param edgeThreshold: float=10, 
            The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).\n
        :param sigma: float=1.6
            The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.\n
        """
        super().__init__()
        self._detector = Detectors.SIFTDetector(
            contrastThreshold=0.09, 
            **kargs
        )
        self._desciptor = Descriptors.SIFTDescriptor()

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

# TODO: 1. self-defined feature class