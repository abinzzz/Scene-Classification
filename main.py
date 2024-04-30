
import os
import cv2
from SIFT import SIFT
import numpy as np
import random
from BoW import BoW
from SVM import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
import time  

dataset_path = "dataset"
categories = os.listdir(dataset_path)
random_seed = 3
random.seed(random_seed)
bags_of_words_count =180



def data_split(data_dir):
    '''
    数据集切分
    '''
    train=[]
    test=[]

    root, dirs,_=next(os.walk(data_dir))
    """
    root: Dataset
    dirs: ['03', '04', '05', '02', '11', '10', '07', '00', '09', '08', '01', '06', '12', '13', '14']
    """
    descriptor_set = np.float32([]).reshape(0, 128)

    label=0
    for dir in dirs:
        files = os.listdir(root + "/" + dir)
       
        for i in range(len(files)):#遍历前150张图片
            if i < 150:
                img = cv2.imread(root + "/" + dir + "/" + files[i])
                sift=SIFT(img,label)
                sift.SIFT_features()
                sift.SIFT_Norm()
                train.append(sift)
                descriptor_set = np.append(descriptor_set, sift.descriptors, axis=0)

            
            else:
                img = cv2.imread(root + "/" + dir + "/" + files[i])
                sift=SIFT(img,label)
                sift.SIFT_features()
                sift.SIFT_Norm()
                test.append(sift)
        label+=1
    

    np.save("npy/descriptor_set.npy", descriptor_set)
    return train, test
                    
                
            

def plotCM(classes, matrix, savename):
    '''
    绘制混淆矩阵
    :param classes: 图片标签名称
    :param matrix: 混淆矩阵
    :param savename: 混淆矩阵保存路径
    :return: None
    '''
    matrix = matrix.astype(np.float64)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix,cmap= LinearSegmentedColormap.from_list('my_cmap', ['white', 'blue']))
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center', fontsize=6)
    ax.set_xticklabels([''] + classes, rotation=90)
    ax.set_yticklabels([''] + classes)
    plt.savefig(savename)
    plt.close(fig)





if __name__ == '__main__':
        start_time = time.time()  # 记录开始时间
        # 检查是否已经存在训练和测试数据文件
        if not os.path.exists("npy/train_set.npy"):
            # 加载数据，并分割成训练集和测试集
            train_samples, test_samples = data_split(dataset_path)
            
            # 初始化词袋模型，生成特征集
            bow = BoW(bags_of_words_count)
            bow.KMeans("npy/descriptor_set.npy", random_seed)
            
            # 使用SPM方法生成训练和测试数据的特征向量
            train_features, train_labels = bow.data2svm(train_samples)
            test_features, test_labels = bow.data2svm(test_samples)
            
            # 保存训练和测试数据
            np.save("npy/train_set.npy", train_features)
            np.save("npy/train_label.npy", train_labels)
            np.save("npy/test_set.npy", test_features)
            np.save("npy/test_label.npy", test_labels)
        else:
            # 加载已保存的训练和测试数据
            train_features = np.load("npy/train_set.npy")
            train_labels = np.load("npy/train_label.npy")
            test_features = np.load("npy/test_set.npy")
            test_labels = np.load("npy/test_label.npy")

        # 使用SVM分类器训练模型并进行预测
        svm_model = train(train_features, train_labels)
        train_predictions = predict(train_features, svm_model)
        test_predictions = predict(test_features, svm_model)

        # 评估训练和测试结果
        _,_,train_classification_report= evaluate(train_predictions, train_labels)
        _,confu_matrix,test_classification_report = evaluate(test_predictions, test_labels)

        # 绘制并保存混淆矩阵图
        categor=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14']
        plotCM(categor, confu_matrix, "output/confusion_matrix.png")
        #print("训练集评估报告：")
        #print(train_classification_report)
        print("测试集评估报告：")
        print(test_classification_report)
        end_time = time.time()
        print(f"程序执行时间：{end_time - start_time:.2f} 秒")


