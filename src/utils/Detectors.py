import cv2
import numpy as np
from abc import ABC, abstractmethod

class Detectors(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def detect(self):
        pass

    def _distribute_keypts(self):
        pass

class Harris:
    def __init__(self) -> None:
        pass

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