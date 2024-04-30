import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import math

class BoW:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = None

    def KMeans(self, feature_path, random_state):
        """
        使用 MiniBatchKMeans 聚类算法生成词袋模型。

        :param feature_path: 特征数据的存储路径。
        :param random_state: 随机数种子，用于聚类算法的初始化。
        :return: 生成的视觉词汇中心矩阵。
        """
        try:
            features = np.load(feature_path)
        except IOError as e:
            raise Exception(f"Error loading features from {feature_path}: {e}")

        np.random.shuffle(features)
        kmeans = MiniBatchKMeans(n_clusters=self.vocab_size, random_state=random_state, batch_size=200).fit(features)
        self.vocab = kmeans.cluster_centers_

        try:
            np.save("npy/Bow.npy", self.vocab)
        except IOError as e:
            raise Exception(f"Error saving vocabulary to npy/Bow.npy: {e}")
        
        # print("self.vocab",self.vocab)
        # print("self.vocab.shape",self.vocab.shape)  
        # print("self.vocab.size",self.vocab.size)

        return self.vocab
    

    def SPM(self, features, keypoints, img_width, img_height):
        """
        使用空间金字塔匹配 (SPM) 算法计算图像的视觉词汇投票向量。

        :param features: 图像的特征点描述符。
        :param keypoints: 图像的关键点列表。
        :param img_width: 图像的宽度。
        :param img_height: 图像的高度。
        :return: 基于 SPM 的视觉词汇投票结果向量。
        """
        num_features = len(features)
        width_step = math.ceil(img_width / 4)
        height_step = math.ceil(img_height / 4)
        level_two_histogram = np.zeros((16, self.vocab_size))
        level_one_histogram = np.zeros((4, self.vocab_size))
        level_zero_histogram = np.zeros((1, self.vocab_size))

        for i in range(num_features):
            feature = features[i]
            x, y = keypoints[i].pt
            index = math.floor(x / width_step) + 4 * math.floor(y / height_step)
            distance = np.linalg.norm(np.tile(feature, (self.vocab_size, 1)) - self.vocab, axis=1)
            closest_center_index = np.argmin(distance)
            level_two_histogram[index][closest_center_index] += 1

        for i in range(4):
            level_one_histogram[i] = np.sum(level_two_histogram[i * 4:(i + 1) * 4], axis=0)

        level_zero_histogram[0] = np.sum(level_one_histogram, axis=0)

        final_vector = np.concatenate((level_zero_histogram * 0.25, level_one_histogram * 0.25, level_two_histogram * 0.5), axis=0)
        return final_vector.flatten()


    def data2svm(self, image_data):
        """
        将图片数据转换为适用于 SVM 训练的格式。

        :param image_data: 包含图片特征点和关键点的数据集。
        :return: SVM 的输入数据和对应的标签。
        """
        dataset = np.float32([])
        labels = []

        for image in image_data:
            spm_vector = self.SPM(image.descriptors, image.keypoints, image.width, image.height)
            dataset = np.vstack([dataset, spm_vector]) if dataset.size else spm_vector
            labels.append(image.label)

        return dataset, np.array(labels)
