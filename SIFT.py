
import cv2
import numpy as np
class SIFT:
    def __init__(self, image, label):
        self.image = image
        self.height, self.width = image.shape[0:2]
        self.label = label
        self.descriptors = None
        self.keypoints = None

    def SIFT_features(self):
        '''
        提取图片的 SIFT 特征点
        '''
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        self.keypoints, self.descriptors = sift.detectAndCompute(gray_image, None)

    def SIFT_Norm(self):
        for i in range(len(self.descriptors)):
            norm = np.linalg.norm(self.descriptors[i])
            if norm > 1:
                self.descriptors[i] /= float(norm)
    

