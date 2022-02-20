import pdb
from typing import Tuple
from weakref import ref

import numpy as np
import cv2

from utils.utils import baseClass
from utils.Detectors import Detectors

class BaseVO(baseClass):
    def __init__(self, K:np.ndarray) -> None:
        super().__init__()
        self._K = K

    def _KLT_tracker(
        self, 
        img0:np.ndarray, img1:np.ndarray, 
        ref_kps:Tuple[cv2.KeyPoint], 
        backtrack_check:bool=True, 
        **KLT_args
    ) -> Tuple[cv2.KeyPoint]:
        """
        :param img0: np.ndarray, original image
        :param img1: np.ndarray, new image to track
        :param ref_kps: keypoints from img0
        :param backtrack_check: bool=True, does the tracker back track the keypoints back to img0 to check its validation 

        :return: n_kp1, selected keypoints for img0
        :return: n_kp2, selected keypoints for img1
        :return: mask, match mask, 
        """
        # forward tracking
        kp1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, ref_kps, None, **KLT_args)
        mask = self._get_in_img_mask(img1, kp1)

        # back tracking
        if backtrack_check:
            kp0, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, kp1, None, **KLT_args)
            dist = abs(ref_kps - kp0).reshape(-1, 2).max(-1)  # Verify the absolute difference between feature points
            back_track_mask = dist < self._extract_param(KLT_args, "back_track_error", 5)
            mask = np.logical_and(mask, back_track_mask)

        valid_kp0 = ref_kps[mask]
        valid_kp1 = kp1[mask]

        return valid_kp0, valid_kp1, mask

    def _get_in_img_mask(self, img1:np.ndarray, kps):
        height, width = img1.shape
        left_mask = kps[:, 0] >= 0
        right_mask = kps[:, 0] <= width-1
        up_mask = kps[:, 1] >= 0
        bottom_mask = kps[:, 1] <= height-1
        width_mask = np.logical_and(left_mask, right_mask)
        height_mask = np.logical_and(up_mask, bottom_mask)
        mask = np.logical_and(width_mask, height_mask)
        return mask

class VisualOdometry(BaseVO):
    def __init__(self, K:np.ndarray) -> None:
        super(VisualOdometry, self).__init__(K)

    # TODO: 0.0 bootstrap_KLT
    def bootstrap_KLT(
        self, 
        img0: np.ndarray, img1:np.ndarray, 
        kps_detector: Detectors, 
        **KLT_args
    ) -> bool:
        """
        :param img0: 
        :param img1:
        :param kps_detector: Detectors
        :param KLT_args: dict, parameters for KLT tracker
        """
        kps0 = kps_detector.detect(img0, 5, 3, 20, **KLT_args)
        _, kp1, mask = self._KLT_tracker(img0, img1, kps0)


    # TODO: 0.1 bootstrap_Matching
    def bootstrap_Matching(self):
        raise NotImplementedError

    # TODO: 0.0 Update KLT
    def update_KLT(self):
        raise NotImplementedError

    # TODO: 0.1 update matching
    def update_matching(self):
        raise NotImplementedError

    

