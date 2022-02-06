from email.policy import default
from importlib.resources import path
import logging
from math import ceil, floor
from turtle import width
from typing import Any, Tuple
import cv2
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum

import pdb

class DET_TYPE(Enum):
    HARRIS = 1 

class Detectors(ABC):
    def __init__(self) -> None:
        super().__init__()

    def detect(
        self, img:np.ndarray, 
        x_cell_num:int=1, y_cell_num:int=1, 
        cell_keypts_max_num:int=None, 
        **kargs
    )->Tuple[cv2.KeyPoint]:
        if len(img.shape)==3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        height, width = img.shape
        block_width, block_height, = 1.*width/x_cell_num, 1.*height/y_cell_num

        kpts_all = ()
        # block_kp_max_num = self._extract_param(kargs, "block_kp_max_num", None)

        for i in range(x_cell_num):
            for j in range(y_cell_num):
                left_bound, right_bound = floor(i*block_width), floor((i+1)*block_width)
                top_bound, bottom_bound = floor(j*block_height), floor((j+1)*block_height)
                patch = img[top_bound:bottom_bound, left_bound:right_bound]
                kpts = self._cell_detect(
                    patch, keypts_max_num=cell_keypts_max_num, **kargs
                )
                if len(kpts)>0:
                    kpts = self._recover_pos(kpts, left_bound, top_bound)
                    kpts_all = kpts_all + kpts

        if len(kpts_all)>0:
            return kpts_all
        else:
            return ()

    @abstractmethod
    def _cell_detect(self, img:np.ndarray, keypts_max_num:int=None, **kargs)->Tuple[cv2.KeyPoint]:
        pass

    def _recover_pos(self, kpts:Tuple[cv2.KeyPoint], x_start:int, y_start:np.ndarray)->np.ndarray:
        for kpt in kpts:
            x,y = kpt.pt
            kpt.pt = (x+x_start, y+y_start)
        return kpts

    def _extract_param(self, param_dict:dict, name:str, default:Any=None, param_type:type=None):
        res = param_dict.get(name)
        if res is None:
            return default
        else:
            if param_type is None or isinstance(res, param_type):
                return res
            else:
                logging.error('result type {} is not of type {}'.format(type(res), param_type))
                raise TypeError

class Harris_Detector(Detectors):
    def __init__(
        self, **kargs
    ) -> None:
        """
        :params block_size: int=3, block size to average the gradient in Harris
        :params sobel_kernel_size:int = 3, sobel kernel size
        :params harris_k:float = 0.04, det(M)-k*tr(M)^2
        :params nms_radius: int=3, nms radius when finding corners
        """
        super().__init__()

        # Harris Parameters
        self._block_size = self._extract_param(kargs, 'block_size', 3, int)
        self._ksize=self._extract_param(kargs, 'sobel_kernel_size', 3, int) 
        self._k=self._extract_param(kargs, 'harris_k', 0.04, float)

        self._nms_radius = self._extract_param(kargs, 'nms_radius', default=3)

    def detect(
        self, 
        img:np.ndarray, 
        x_cell_num:int=1, y_cell_num:int=1, 
        cell_keypts_max_num:int=None, 
        **kargs
    ) -> cv2.KeyPoint:   
        """
        Distribute the keypoints enenly through out the image. 
        First cutting the image into blocks. Do Harris Detect in every block.
        Finally stack all the result together

        :params img: gray scale image
        :params x_cell_num: int = 1, the number of columns when cutting the image into blocks
        :params y_cell_num: int = 1, the number of lines when cutting the image into blocks
        :params cell_keypts_max_num: int=None, maximum keypoints number in each cell, None for infinity
        :params harris_threshold: float=5e-3, threshold for harris corner response function

        :return: tuple of n keypoints 
        """
        return super().detect(img, x_cell_num, y_cell_num, cell_keypts_max_num, **kargs)


    def _cell_detect(
        self, 
        img: np.ndarray, 
        keypts_max_num: int=None, 
        **kargs
    )->Tuple[cv2.KeyPoint]:
        """
        :params img: gray scale image
        :params harris_threshold: float = 5e-3,
        :params keypts_max_num: int=None, maximum keypoints number in each cell, None for infinity

        :return: (n*2) ndarray for n keypoints in image coordinate
        """
        threshold = self._extract_param(kargs, 'harris_threshold', 5e-3, float)

        corner_response_map = cv2.cornerHarris(
            img, blockSize=self._block_size, 
            ksize=self._ksize, 
            k=self._k
        )

        corner_response_map[corner_response_map<0] = 0
        kps = self._get_kps(
            corner_response_map, 
            response_threshold=threshold, 
            kp_max_num=keypts_max_num)

        return kps

    def _get_kps(
        self, 
        harris_response_map:np.ndarray, 
        response_threshold:float, kp_max_num:int=None, 
    ) -> Tuple[cv2.KeyPoint]:
        # init
        key_pts_list = []
        scores = np.copy(harris_response_map)
        scores[scores<response_threshold] = 0

        while True:
            if scores.max() == 0:
                break
            if kp_max_num is not None and len(key_pts_list)>=kp_max_num:
                break
            kp = np.unravel_index(scores.argmax(), scores.shape)
            key_pts_list.append((kp[1], kp[0]))
            scores = self._supress(scores, kp, self._nms_radius)
        kps_np = np.array(key_pts_list, dtype=np.float64)

        return cv2.KeyPoint_convert(kps_np)

    def _supress(self, scores:np.ndarray, kp:Tuple[int], radius:int=3)->np.ndarray:
        height, width = scores.shape
        left_bound = max(0, kp[1]-radius)
        top_bound = max(0, kp[0]-radius)
        right_bound = min(width, kp[1]+radius+1)
        bottom_bound = min(height, kp[0]+radius+1)

        scores[top_bound:bottom_bound, left_bound:right_bound] = 0

        return scores

class SIFT_Detector(Detectors):
    def __init__(self, **kargs) -> None:
        """
        :param nOctaveLayers: int=3, 
            The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
        :param contrastThreshold: float=0.04, 
            The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.
            note The contrast threshold will be divided by nOctaveLayers when the filtering is applied. When nOctaveLayers is set to default and if you want to use the value used in D. Lowe paper, 0.03, set this argument to 0.09.
        :param edgeThreshold: float=10, 
            The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
        :param sigma: float=1.6
            The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
        """
        super().__init__()
        # params

        self._sift_detector = cv2.SIFT_create(
            # nfeatures=self._extract_param(kargs, "nfeatures", 100, int), 
            nOctaveLayers=self._extract_param(kargs, "nOctaveLayers", 3, int), 
            contrastThreshold=self._extract_param(kargs, "contrastThreshold", 0.04, float), 
            edgeThreshold=self._extract_param(kargs, "edgeThreshold", 10, float),
            sigma=self._extract_param(kargs, "sigma", 1.6, float),
        )
        cv2.SIFT_create()

    def _cell_detect(self, img:np.ndarray, keypts_max_num:int=None, **kargs)->Tuple[cv2.KeyPoint]:
        kps = self._sift_detector.detect(img, None)
        return kps[:keypts_max_num]

    # def compute(
    #     self, 
    #     img:np.ndarray, 
    #     keypoints:Tuple[cv2.KeyPoint]
    # ) -> np.ndarray:
    #     descriptors = self._sift_detector.compute(img, keypoints, None)
    #     return descriptors[1]


DETECTORS = {
    DET_TYPE.HARRIS: Harris_Detector
}

