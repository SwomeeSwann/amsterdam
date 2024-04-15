import torch
import os
import cv2 as cv
import random

class SiameseData():
    def __init__(self, img_dir) -> None:
        self.dir = img_dir
        self.trainset = ["img", "img", "img"]
        

    def __getitem__(self, index):
        img_path = os.path.join(self.dir, self.trainset[index])
        
        img = cv.imread(img_path)
        
        grey_img = cv.cvtColor(img, cv.COLOR_BAYER_BG2GRAY)
        
        return grey_img



class SiameseCat():
    def __init__(self) -> None:
        super(SiameseCat, self).__init__()
        
        self.nn1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, kernel_size=(3,3))
        )