## 作业说明
**任务:** 编写一个图像分类系统，能够对输入图像进行类别预测。具体的说，利用数据库的 2250 张训练样本进行训练；对测试集中的 2235 张样本进行预测。

**数据库说明:** `scene_categories` 数据集包含 15 个类别（文件夹名就是类别名），每个类中编号前 150 号的样本作为训练样本，15 个类一共 2250 张训练样本；剩下的样本构成测试集合。

**使用知识点:** SIFT 特征、Kmeans、词袋表示、支撑向量机


**数据集下载地址:** [15-Scene_Image_Dataset](https://figshare.com/articles/dataset/15-Scene_Image_Dataset/7007177)

**参考论文地址:** [CVPR 2006 - Scene Recognition with Bag-of-Words](http://people.csail.mit.edu/torralba/courses/6.870/papers/cvpr06b.pdf)


## 算法流程

---- 数据准备与特征提取 ----

1. **提取数据集中的样本，并划分训练集和测试集**：
   - 从完整的数据集中随机选取样本。
   - 将样本分为两部分：训练集和测试集。

2. **特征提取与词典生成**：
   - **提取SIFT特征**：对训练集中的所有图片提取SIFT特征点，并对这些特征点向量进行归一化处理。
   - **聚类生成视觉词汇**：使用K-means算法将所有SIFT特征点分为n类，每类的中心点形成视觉词汇。
   - **生成词袋（字典）**：将n类特征点的中心点作为视觉词汇，创建一个词袋模型。

---- 图片表示 ----

3. **使用SPM算法生成图片的特征向量**：
   - **多尺度划分**：将图片分为3种尺度，具体为1x1、2x2、4x4大小的区块。
   - **特征直方图统计**：统计每个尺度下的特征直方图。
   - **合并特征直方图**：将不同尺度下的特征直方图合并，形成一个21（1+4+16）*n维的特征向量。

---- 执行分类 ----

4. **训练分类模型**：
   - **模型训练**：使用支持向量机（SVM）算法，以训练集中的图片特征向量作为数据集进行模型训练。

5. **测试与评估**：
   - **数据处理**：对测试集中的图片进行与训练集相同的数据处理操作。
   - **模型预测**：使用训练好的支持向量机模型对测试集图片的类别进行预测。
   - **评估结果**：评估预测结果的准确性，并生成分类报告和混淆矩阵。



## 数据集
```
├── Dataset
│   ├── 00
│   ├── 01
│   ├── 02
│   ├── 03
│   ├── 04
│   ├── 05
│   ├── 06
│   ├── 07
│   ├── 08
│   ├── 09
│   ├── 10
│   ├── 11
│   ├── 12
│   ├── 13
│   └── 14
```



## 参考链接
- [SIFT](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html)