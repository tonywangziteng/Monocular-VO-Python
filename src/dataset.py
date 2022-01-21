from os import read
import cv2
import os.path as osp
from abc import ABC, abstractmethod
import numpy as np
import os

import pdb

class Dataset(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_next_image(self) -> np.ndarray:
        pass
    

class Parking_dataset(Dataset):
    def __init__(self, img_ptr=0) -> None:
        super(Parking_dataset, self).__init__()
        self.__dataset_root_path = "../parking/"
        self.__img_dir_path = osp.join(self.__dataset_root_path, "images")
        
        # load K matrix
        K_path = osp.join(self.__dataset_root_path, 'K.txt')
        with open(K_path) as K_file:
            K_str = K_file.read()
        self.__K = np.fromstring(K_str, sep=',').reshape(3, 3)

        self.__img_namelist = os.listdir(self.__img_dir_path)
        # only images
        self.__img_namelist = \
            [file_name for file_name in self.__img_namelist if file_name.endswith('png')]
        self.__img_namelist.sort()
        self.__img_num = len(self.__img_namelist)
        self.__img_ptr = img_ptr

    def get_next_image(self) -> np.ndarray:
        if self.__img_ptr == self.__img_num:
            return None, None

        img_name = self.__img_namelist[self.__img_ptr]
        img_full_path = osp.join(self.__img_dir_path, self.__img_namelist[self.__img_ptr])
        rgb_img = cv2.imread(img_full_path)
        self.__img_ptr+=1

        return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY), rgb_img, img_name

    def get_img(self, img_ptr:int) -> np.ndarray:
        assert (img_ptr>=0 and img_ptr<self.__img_num), "img pointer out of range"

        img_name = self.__img_namelist[img_ptr]
        img_full_path = osp.join(self.__img_dir_path, self.__img_namelist[self.__img_ptr])
        rgb_img = cv2.imread(img_full_path)

        return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY), img_name

        
    @property
    def K(self) -> np.ndarray:
        return self.__K

class KITTI_dataset(Dataset):
    def __init__(self, img_ptr=0) -> None:
        super(KITTI_dataset, self).__init__()
        self._dataset_root_path = "../kitti/05/"
        self._img_dir_path = osp.join(self._dataset_root_path, "image_0")
        
        # load K matrix
        self._K=np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                        [0, 7.188560000000e+02, 1.852157000000e+02],
                        [0 ,0 ,1]])

        self._img_namelist = os.listdir(self._img_dir_path)
        # only images
        self._img_namelist = \
            [file_name for file_name in self._img_namelist if file_name.endswith('png')]
        self._img_namelist.sort()
        self._img_num = len(self._img_namelist)
        self._img_ptr = img_ptr

    def get_next_image(self) -> np.ndarray:
        if self._img_ptr == self._img_num:
            return None, None

        img_name = self._img_namelist[self._img_ptr]
        img_full_path = osp.join(self._img_dir_path, self._img_namelist[self._img_ptr])
        rgb_img = cv2.imread(img_full_path)
        self._img_ptr+=1

        return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY), rgb_img, img_name

    def set_ptr(self, ptr_idx:int):
        self._img_ptr = ptr_idx

    def get_img(self, img_ptr:int) -> np.ndarray:
        assert (img_ptr>=0 and img_ptr<self._img_num), "img pointer out of range"

        img_name = self._img_namelist[img_ptr]
        img_full_path = osp.join(self._img_dir_path, self._img_namelist[self._img_ptr])
        rgb_img = cv2.imread(img_full_path)

        return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY), img_name
        
    @property
    def K(self) -> np.ndarray:
        return self._K

    @property
    def index(self) -> np.ndarray:
        return self._img_ptr


datasets = {
    "KITTI": KITTI_dataset, 
    "parking": Parking_dataset
}
