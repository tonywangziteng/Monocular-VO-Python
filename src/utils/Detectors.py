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

    @abstractmethod
    def detect(
        self, 
        img:np.ndarray, 
    ):
        if len(img.shape)==3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _distribute_keypts(self):
        pass

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

    def _cell_detect(
        self, 
        img: np.ndarray, 
        threshold: float=5e-3, 
        keypts_max_num: int=None
    )->np.ndarray:
        """
        :params img: gray scale image
        :params harris_threshold: float = 5e-3,
        :params keypts_max_num: int=None, maximum keypoints number in each cell, None for infinity

        :return: (n*2) ndarray for n keypoints in image coordinate
        """
        super().detect(img)

        corner_response_map = cv2.cornerHarris(
            img, blockSize=self._block_size, 
            ksize=self._ksize, 
            k=self._k
        )

        corner_response_map[corner_response_map<0] = 0
        kps = self._get_kps(
            corner_response_map, 
            response_threshold=threshold, 
            nms_radius=self._nms_radius,
            kp_max_num=keypts_max_num)

        return kps

    def detect(
        self, 
        img:np.ndarray, 
        harris_threshold:float=5e-3,
        x_num:int=1, y_num:int=1, 
        cell_keypts_max_num:int=None 
    ) -> cv2.KeyPoint:   
        """
        Distribute the keypoints enenly through out the image. 
        First cutting the image into blocks. Do Harris Detect in every block.
        Finally stack all the result together

        :params img: gray scale image
        :params harris_threshold: float=5e-3, threshold for harris corner response function
        :params x_num: int = 1, the number of columns when cutting the image into blocks
        :params y_num: int = 1, the number of lines when cutting the image into blocks
        :params cell_keypts_max_num: int=None, maximum keypoints number in each cell, None for infinity

        :return: (n*2) ndarray for n keypoints in image coordinate
        """
        height, width = img.shape
        block_width, block_height, = 1.*width/x_num, 1.*height/y_num

        kpts_list = []
        # block_kp_max_num = self._extract_param(kargs, "block_kp_max_num", None)

        for i in range(x_num):
            for j in range(y_num):
                left_bound, right_bound = floor(i*block_width), floor((i+1)*block_width)
                top_bound, bottom_bound = floor(j*block_height), floor((j+1)*block_height)
                patch = img[top_bound:bottom_bound, left_bound:right_bound]
                kpts = self._cell_detect(
                    patch, harris_threshold, keypts_max_num=cell_keypts_max_num
                )
                if len(kpts)>0:
                    kpts = self._recover_pos(kpts, left_bound, top_bound)
                    kpts_list.append(kpts)

        if len(kpts_list)>0:
            return cv2.KeyPoint_convert(np.concatenate(kpts_list, axis=0)) 
        else:
            return ()
    
    def _recover_pos(self, kpts:np.ndarray, x_start:int, y_start:np.ndarray)->np.ndarray:
        kpts[:,0] = kpts[:,0]+x_start
        kpts[:,1] = kpts[:,1]+y_start
        return kpts

    def _get_kps(
        self, 
        harris_response_map:np.ndarray, 
        response_threshold:float, nms_radius:int=None, kp_max_num:int=None, 
    ) -> np.ndarray:
        # init
        key_pts_list = []
        scores = np.copy(harris_response_map)
        scores[scores<response_threshold] = 0

        # set default value
        if nms_radius is None: nms_radius = 3

        while True:
            if scores.max() == 0:
                break
            if kp_max_num is not None and len(key_pts_list)>=kp_max_num:
                break
            kp = np.unravel_index(scores.argmax(), scores.shape)
            key_pts_list.append((kp[1], kp[0]))
            scores = self._supress(scores, kp, nms_radius)
        return np.array(key_pts_list, dtype=np.float64)

    def _supress(self, scores:np.ndarray, kp:Tuple[int], radius:int=3)->np.ndarray:
        height, width = scores.shape
        left_bound = max(0, kp[1]-radius)
        top_bound = max(0, kp[0]-radius)
        right_bound = min(width, kp[1]+radius+1)
        bottom_bound = min(height, kp[0]+radius+1)

        scores[top_bound:bottom_bound, left_bound:right_bound] = 0

        return scores



# TODO: SIFT Detector and Descriptor
# 看看cv2.Keypoint 里面有没有scale 和 旋转信息
class SIFT_Detector(Detectors):
    def __init__(self, **kargs) -> None:
        """
        :param nfeatures: int=100, The number of best features to retain. The features are ranked by their scores
        """
        super().__init__()
        # params
        self._sift_detector = cv2.SIFT_create(
            nfeatures=self._extract_param(kargs, "nfeatures", 100, int), 
            # nOctaveLayers=self._extract_param(kargs, "nOctaveLayers", 100, int), 
            # contrastThreshold=self._extract_param(kargs, "contrastThreshold", 100, int), 
            # edgeThreshold=self._extract_param(kargs, "edgeThreshold", 100, int),
            # sigma=self._extract_param(kargs, "sigma", 100, int),
        )

    def detect(self, img:np.ndarray):
        descriptor = cv2.SiftDescriptorExtractor
        return self._sift_detector.detect(img, None)

DETECTORS = {
    DET_TYPE.HARRIS: Harris_Detector
}

