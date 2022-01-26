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
        self, 
    ) -> None:
        super().__init__()

    def _block_detect(
        self, 
        img: np.ndarray, 
        **kargs
    )->np.ndarray:
        """
        :params img: gray scale image
        :params harris_threshold:float = 5e-3,
        :params **kparams:
            harris_threshold:float = 5e-3, 
            block_kp_max_num: int, max keypoint number to return, default infinity
            nms_radius: int, nms radius when selecting keypoints
            block_size: int=3, block size to average the gradient
            sobel_kernel_size:int = 3, 
            harris_k:float = 0.04, 

        :return: (n*2) ndarray for n keypoints in image coordinate
        """
        super().detect(img)

        corner_response_map = cv2.cornerHarris(
            img, blockSize=self._extract_param(kargs, 'block_size', 3, int), 
            ksize=self._extract_param(kargs, 'sobel_kernel_size', 3, int), 
            k=self._extract_param(kargs, 'harris_k', 0.04, float)
        )

        corner_response_map[corner_response_map<0] = 0
        kps = self._get_kps(
            corner_response_map, 
            response_threshold=self._extract_param(kargs, 'harris_threshold', 5e-3), 
            nms_radius=self._extract_param(kargs, 'nms_radius', default=3),
            kp_max_num=self._extract_param(kargs, 'block_kp_max_num')
        )

        return kps

    # TODO: 参数改到初始化里面
    def detect(
        self, 
        img:np.ndarray, 
        x_num:int=1, y_num:int=1,  
        **kargs
    ) -> cv2.KeyPoint:   
        """
        Distribute the keypoints enenly through out the image. 
        First cutting the image into blocks. Do Harris Detect in every block.
        Finally stack all the result together

        :params img: gray scale image
        :params x_num: int = 1, the number of columns when cutting the image into blocks
        :params y_num: int = 1, the number of lines when cutting the image into blocks
        :params **kparams:
            harris_threshold:float = 5e-3, 
            block_kp_max_num: int, max keypoint number to return, default infinity
            nms_radius: int, nms radius when selecting keypoints
            block_size: int=3, block size to average the gradient
            sobel_kernel_size:int = 3, 
            harris_k:float = 0.04, 

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
                print(left_bound, right_bound)
                patch = img[top_bound:bottom_bound, left_bound:right_bound]
                kpts = self._block_detect(patch, **kargs)
                if len(kpts)>0:
                    print(len(kpts))
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



# TODO: 完成tow-stage的SIFT
# 看看cv2.Keypoint 里面有没有scale 和 旋转信息
class SIFT_Detector(Detectors):
    def __init__(self) -> None:
        super().__init__()

DETECTORS = {
    DET_TYPE.HARRIS: Harris_Detector
}

