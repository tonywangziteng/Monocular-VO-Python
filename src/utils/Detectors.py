from typing import Tuple
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

class Harris_Detector(Detectors):
    def __init__(
        self, 
    ) -> None:
        super().__init__()

    # def detect(self, img: np.ndarray):
    #     super().detect(img)

    def detect(
        self, 
        img: np.ndarray, 
        nms_size:int = 3, 
        sobel_kernel_size:int = 3, 
        harris_threshold:float = 2e-2,
        harris_k:float = 0.04, 
        **kparams
    ):
        super().detect(img)
        kp_max_num = kparams.get('kp_max_num')

        corner_response_map = cv2.cornerHarris(
            img, blockSize=nms_size, ksize=sobel_kernel_size, k=harris_k
        )
        corner_response_map[corner_response_map<0] = 0
        kps = self._get_kps(
            corner_response_map, response_threshold=4e-2, kp_max_num=kp_max_num
        )
        # TODO: distribute keypoints
        pdb.set_trace()

    def _get_kps(
        self, 
        harris_response_map:np.ndarray, 
        response_threshold:float, nms_radius:int=3, kp_max_num:int=None, 
    ) -> np.ndarray:
        key_pts_list = []
        scores = np.copy(harris_response_map)
        scores[scores<response_threshold] = 0

        while True:
            print(scores.max())
            if scores.max() == 0:
                break
            if kp_max_num is not None and len(key_pts_list)>=kp_max_num:
                break
            kp = np.unravel_index(scores.argmax(), scores.shape)
            key_pts_list.append(kp)
            scores = self._supress(scores, kp, nms_radius)
        np.array(key_pts_list, dtype=np.float64)

    def _supress(self, scores:np.ndarray, kp:Tuple[int], radius:int=3)->np.ndarray:
        height, width = scores.shape
        left_bound = max(0, kp[1]-radius)
        top_bound = max(0, kp[0]-radius)
        right_bound = min(width-1, kp[1]+radius)
        bottom_bound = min(height-1, kp[0]+radius)

        scores[top_bound:bottom_bound, left_bound:right_bound] = 0
        return scores



class SIFT_Detector(Detectors):
    def __init__(self) -> None:
        super().__init__()

DETECTORS = {
    DET_TYPE.HARRIS: Harris_Detector
}



# class Harris:

#     def __init__(self,harris_patch_size,harris_kappa,
#                  query_keypoint_num,
#                  nonmaximum_supression_radius):

#         self.harris_patch_size=harris_patch_size
#         self.harris_kappa=harris_kappa
#         self.kp_num=query_keypoint_num
#         self.suppresion_radius=nonmaximum_supression_radius
#         self.split_num=4





#     def calculate_Harris(self,img):
#         # I_x=cv2.Sobel(self.img,ddepth=cv2.CV_8U,dx=1,dy=0,ksize=3)
#         # I_y=cv2.Sobel(self.img,ddepth=cv2.CV_8U,dx=0,dy=1,ksize=3)
#         I_x = convolve(img.astype('float'), [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#         I_y = convolve(img.astype('float'), [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

#         I_x_2 = np.square(I_x)
#         I_y_2 = np.square(I_y)

#         I_x_mul_I_y = I_x * I_y

#         sigma_I_x_2 = convolve(I_x_2, np.ones(shape=(self.harris_patch_size, self.harris_patch_size)))
#         sigma_I_y_2 = convolve(I_y_2, np.ones(shape=(self.harris_patch_size, self.harris_patch_size)))
#         sigma_I_x_mul_I_y = convolve(I_x_mul_I_y, np.ones(shape=((self.harris_patch_size, self.harris_patch_size))))

#         det = sigma_I_x_2 * sigma_I_y_2 - np.square(sigma_I_x_mul_I_y)
#         trace = sigma_I_x_2 + sigma_I_y_2

#         tmp = det - self.harris_kappa * np.square(trace)
#         tmp[tmp < 0] = 0
#         self.Harris = tmp



#     def select_keypoints(self,img):
#         self.calculate_Harris(img)
#         keypoint_coord = np.zeros((self.kp_num, 2))
#         # scores=np.ones(shape=Harris.shape,dtype=float)
#         # np.copyto(scores,Harris)
#         scores = np.copy(self.Harris)
#         scores = np.pad(scores, self.suppresion_radius)

#         h, w = self.Harris.shape
#         nonmax_radius=self.suppresion_radius


#         for i in range(0, self.kp_num):
#             max_index = np.argmax(scores)

#             max_h, max_w = np.unravel_index(max_index, (h + 2 * nonmax_radius, w + 2 * nonmax_radius))
#             keypoint_coord[i, :] = [max_w - nonmax_radius, max_h - nonmax_radius]
#             scores[max_h - nonmax_radius:max_h + nonmax_radius, max_w - nonmax_radius:max_w + nonmax_radius] = 0

#         keypoint_coord = keypoint_coord.astype(int)

#         return keypoint_coord

#     def distribute_keypoints(self,img):
#         h,w=img.shape
#         split_h=8
#         split_w=8

#         h_list=np.linspace(0,h,split_h+1)[:-1].astype(int)
#         w_list=np.linspace(0,w,split_w+1)[:-1].astype(int)

#         delta_h=int(h/split_h)
#         delta_w=int(w/split_w)


#         keypoint=np.zeros((0,2))
#         for i in range(split_h):
#             for j in range(split_w):
#                 keypoint_sub=self.select_keypoints(img[h_list[i]:h_list[i]+delta_h,w_list[j]:w_list[j]+delta_w])\
#                              +np.array([[w_list[j],h_list[i]]])
#                 keypoint=np.vstack((keypoint,keypoint_sub))

#         return keypoint