# coding:UTF-8

import numpy as np
import math
import cv2
import time

from collections import deque

class ConDBSCAN():
    '''基于卷积实现的DBSCAN算法，仅用于图像分割'''
    def __init__(self, kernel, minpts):
        self.kernel = kernel                                         # 卷积核，三维np数组
        self.minpts = minpts                                         # 阈值参数
        self.kernel_radius = math.ceil((kernel.shape[0] - 1) / 2)    # 卷积核半径

    def fit_test(self, data, dealed):
        '''
        1. 快速三维卷积计算核心对象得到 convoluted_result
        2. 通过阈值参数 minpts 筛选 convoluted_result 中的核心对象，保存在 core_objects 中
        3. 迭代生成聚类簇，聚类结果保存在 dealed 中，簇标签保存在 labels 中
        '''
        convoluted_result = self.__quick_convolute_3d(data)
        core_objects = self.__get_core_objects(convoluted_result, data)
        print('核心对象个数： ' + str(len(core_objects)))

        visits = deque()
        labels = set()          # 保存簇标签
        tag = 0                 # 簇标识
        for core in core_objects.keys():
            if core_objects[core] == 1:
                continue
            visits.append(core)
            core_objects[core] = 1
            last = (0, 0, 0)
            print(core)
            while len(visits) != 0:
                sample = visits.popleft()
                dealed[sample] = tag
                if core_objects.get(sample, -2) != -2:
                    start = [sample[i] - self.kernel.shape[i] // 2 for i in range(3)]
                    start = [x if x >= 0 else 0 for x in start]
                    end = [sample[i] + self.kernel.shape[i] // 2 for i in range(3)]
                    end = [x if x <= 255 else 255 for x in end]
                    for i in range(start[0], end[0] + 1):
                        for j in range(start[1], end[1] + 1):
                            for k in range(start[2], end[2] + 1):
                                if i <= last[0] and j <= last[1] and k <= last[2]:
                                    continue
                                # 如果是样本点，并且之前未曾标记
                                if dealed.get((i, j, k), -2) == -1:
                                    if core_objects.get((i, j, k), -2) == -1:
                                        visits.append((i, j, k))
                                        core_objects[(i, j, k)] = 1
                                        # print(str((i, j, k)) + ' ' + str(tag))
                                    dealed[(i, j, k)] = tag
                                    print('%d %d %d = %d' % (i, j, k, tag))
                    last = sample
            labels.add(tag)
            tag += 1
        return dealed, labels

    def __calculate_local_sum(self, input, kernel_radius):
        rows, cols = input.shape
        if rows < kernel_radius:
            print('error: kernel size is too biger!')
            return -1

        local_sum = np.zeros((rows, cols), dtype=np.int)
        # 初始化，计算第0行local_sum_0
        for i in range(kernel_radius + 1):
            for j in range(cols):
                local_sum[0, j] += input[i, j]
        for i in range(1, rows):
            if i <= kernel_radius:
                for j in range(cols):
                    local_sum[i, j] = local_sum[i - 1, j] + input[i + kernel_radius, j]
            elif i < rows - kernel_radius:
                for j in range(cols):
                    local_sum[i, j] = local_sum[i - 1, j] + input[i + kernel_radius, j] - input[i - kernel_radius - 1, j]
            else:
                for j in range(cols):
                    local_sum[i, j] = local_sum[i - 1, j] - input[i - kernel_radius - 1, j]
        return local_sum

    def __calculate_convolute_value(self, input, local_sum):
        r = self.kernel_radius
        rows, cols = input.shape
        if cols < r:
            print('error: kernel size is too biger!')
            return -1

        result = np.zeros((rows, cols), dtype=np.int)
        # 初始化，计算result第0列结果
        for i in range(rows):
            for j in range(r + 1):
                result[i, 0] += local_sum[i, j]
        for i in range(rows):
            for j in range(1, cols):
                if j <= r:
                    result[i, j] = result[i, j - 1] + local_sum[i, j + r]
                elif j < cols - r:
                    result[i, j] = result[i, j - 1] + local_sum[i, j + r] - local_sum[i, j - r - 1]
                else:
                    result[i, j] = result[i, j - 1] - local_sum[i, j - r - 1]
        return result

    def __quick_convolute_2d(self, input):
        '''
        * description: 二维卷积
        * input:
            input: 输入矩阵
        * output:
            result: 二维卷积结果，是一个二维数组
        '''
        local_sum = self.__calculate_local_sum(input, self.kernel_radius)
        result = self.__calculate_convolute_value(input, local_sum)
        return result

    def __quick_convolute_3d(self, data):
        '''
        * description: 基于线性规划优化三维卷积
        * input:
            data: 表示图像颜色信息的三维矩阵
        * output:
            c3d_result: 三维卷积结果，是一个三维数组
        '''
        r, c, h = data.shape
        c2d_result = np.zeros((r, c, h), dtype=np.int)
        # 首先对每一层进行卷积
        for i in range(r):
            c2d_result[i, :, :] = self.__quick_convolute_2d(data[i, :, :])
        # 对卷积后的矩阵换一个维度再次进行卷积
        c3d_result = np.zeros((r, c, h), dtype=np.int)
        for j in range(c):
            c3d_result[:, j, :] = self.__calculate_local_sum(c2d_result[:, j, :], self.kernel_radius)
        return c3d_result

    def __get_core_objects(self, convoluted_result, data):
        '''
        * description: 根据阈值参数 minpts 筛选核心对象
        * input:
            convoluted_result: 三维卷积结果
            data: 表示图像颜色信息的三维矩阵
        * output:
            core_objects: 保存核心对象的字典
        '''
        core_objects = {}                    # 保存核心对象
        for i in range(256):
            for j in range(256):
                for k in range(256):
                    if convoluted_result[i, j, k] > self.minpts and data[i, j, k] != 0:
                        core_objects[(i, j, k)] = -1
        return core_objects

    def image_to_3DMatrix(self, img):
        '''
        * description: 将图像映射向RGB颜色空间
        * input:
            img: 输入图像
        * output:
            color_matrix: 表示图像颜色信息的三维矩阵
            color_list: 图像中包含的颜色列表
        '''
        color_matrix = np.zeros((256, 256, 256))
        dealed = {}
        rows, cols, channels = img.shape
        for i in range(rows):
            for j in range(cols):
                b, g, r = img[i, j]
                color_matrix[b, g, r] = 1
                dealed[(b, g, r)] = -1
        return color_matrix, dealed

def my_show(win_name, src):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, src)

if __name__ == '__main__':
    '''
    功能测试
    '''
    start = time.time()
    # img = cv2.imread(r'E:\MachineLearning\ImageSegmentation_Cluster\capsule_images\origin\capsule_1.bmp')
    img = cv2.imread(r'E:\MachineLearning\ImageSegmentation_Cluster\capsule_images\xdpi\capsule_2.bmp')
    conDB = ConDBSCAN(np.ones((7, 7, 7)), 60)
    data, dealed = conDB.image_to_3DMatrix(img)
    print('数据集大小： ' + str(len(dealed)))
    clusters, labels = conDB.fit_test(data, dealed)
    end = time.time()
    print('程序运行时间：%f s' % (end - start))

    print('分类簇个数：%d' % len(labels))

    rows, cols, channels = img.shape
    for tag in labels:
        segmented_image = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                pixel = tuple(img[i, j])
                if clusters[pixel] != tag:
                    segmented_image[i, j] = 255
        my_show('target_' + str(tag), segmented_image)
        cv2.waitKey(0)
    '''
    with open('result.txt', 'w') as file:
        for res in dealed.keys():
            file.write(str(res) + ' ' + str(dealed[res]) + '\n')
    '''
