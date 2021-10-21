# coding:UTF-8

import numpy as np
import math
import cv2
import time

from collections import deque

color_list = np.array([[32, 178, 170], [240, 248, 255], [135, 206, 250], [138, 43, 226], [22, 184, 135], [165, 42, 42],
                       [60, 170, 113], [255, 140, 0], [173, 255, 47], [0, 128, 128], [166, 127, 120], [50, 67, 95],
                       [143, 134, 129], [225, 220, 217], [166, 127, 120], [145, 92, 76], [47, 24, 18], [64, 104, 106],
                       [250, 240, 230], [255, 228, 225], [100, 149, 237], [176, 196, 222], [46, 139, 87], [255, 174, 185],
                       [139, 58, 98], [193, 205, 205], [99, 184, 255], [141, 182, 205], [171, 130, 255], [139, 34, 82],
                       [139, 71, 93], [205, 179, 139], [255, 222, 173], [255, 99, 71], [205, 91, 69], [255, 165, 0], [139, 87, 66],
                       [205, 102, 29], [238, 154, 73], [139, 126, 102], [255, 231, 186], [238, 197, 145], [238, 121, 66],
                       [139, 101, 8], [238, 173, 14], [139, 105, 20], [238, 238, 0], [205, 205, 180], [205, 190, 112],
                       [162, 205, 90], [105, 139, 34], [192, 255, 62]], np.uint8)

class ConDBSCAN():
    '''基于卷积实现的DBSCAN算法，时间复杂度O(n)，空间复杂度O(1)，仅用于图像分割'''
    def __init__(self, kernel, minpts):
        self.channels = 3
        self.depth_bits = 8
        self.kernel = kernel                                         # 卷积核，三维np数组
        self.minpts = minpts                                         # 阈值参数
        self.kernel_radius = [math.ceil((kernel.shape[i] - 1) / 2) for i in range(self.channels)]    # 卷积核半径

    def fit(self, data, dealed):
        '''
        1. 快速三维卷积计算核心对象得到 convoluted_result
        2. 通过阈值参数 minpts 筛选 convoluted_result 中的核心对象，保存在 core_objects 中
        3. 迭代生成聚类簇，聚类结果保存在 dealed 中，簇标签保存在 labels 中
        '''
        max_val = int(math.pow(2, self.depth_bits)) - 1
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
            while len(visits) != 0:
                sample = visits.popleft()
                dealed[sample] = tag
                if core_objects.get(sample, -2) != -2:
                    start = [sample[i] - self.kernel.shape[i] // 2 for i in range(self.channels)]
                    start = [x if x >= 0 else 0 for x in start]
                    end = [sample[i] + self.kernel.shape[i] // 2 for i in range(self.channels)]
                    end = [x if x <= max_val else max_val for x in end]
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
                                    # print('%d %d %d = %d' % (i, j, k, tag))
                    last = sample
            labels.add(tag)
            tag += 1
        return dealed, labels

    def __convolution_x(self, input, radius):
        '''
        * description: X方向卷积
        * input:
            input: 输入矩阵（二维数组）
            radius: 卷积核尺寸
        * output:
            local_sum: X方向卷积结果（二维数组）
        '''
        rows, cols = input.shape
        if rows < radius:
            print('error: kernel size is too biger!')
            return -1

        local_sum = np.zeros((rows, cols), dtype=np.int)
        # 初始化，计算第0行local_sum_0
        for i in range(radius + 1):
            for j in range(cols):
                local_sum[0, j] += input[i, j]

        for i in range(1, rows):
            if i <= radius:
                for j in range(cols):
                    local_sum[i, j] = local_sum[i - 1, j] + input[i + radius, j]
            elif i < rows - radius:
                for j in range(cols):
                    local_sum[i, j] = local_sum[i - 1, j] + input[i + radius, j] - input[i - radius - 1, j]
            else:
                for j in range(cols):
                    local_sum[i, j] = local_sum[i - 1, j] - input[i - radius - 1, j]
        return local_sum

    def __convolution_y(self, input):
        '''
        * description: Y方向卷积
        * input:
            input: 输入矩阵（二维数组）
        * output:
            local_sum: Y方向卷积结果（二维数组）
        '''
        radius_y = self.kernel_radius[1]               # Y方向卷积核尺寸
        rows, cols = input.shape
        if rows < radius_y:
            print('error: kernel size is too biger!')
            return -1

        local_sum = np.zeros((rows, cols), dtype=np.int)
        # 初始化，计算第0行local_sum_0
        for i in range(rows):
            for j in range(radius_y + 1):
                local_sum[i, 0] += input[i, j]

        for i in range(rows):
            for j in range(i, cols):
                if j <= radius_y:
                    local_sum[i, j] = local_sum[i, j - 1] + input[i, j + radius_y]
                elif j < rows - radius_y:
                    local_sum[i, j] = local_sum[i, j - 1] + input[i, j + radius_y] - input[i, j - radius_y - 1]
                else:
                    local_sum[i, j] = local_sum[i, j - 1] - input[i, j - radius_y - 1]
        return local_sum

    def __quick_convolute_2d(self, input):
        '''
        * description: 二维卷积，先进行X方向卷积，再进行Y方向卷积（调换方向结果相同）
        * input:
            input: 输入矩阵
        * output:
            result: 二维卷积结果，是一个二维数组
        '''
        local_sum = self.__convolution_x(input, self.kernel_radius[0])
        c2d_result = self.__convolution_y(local_sum)
        return c2d_result

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
        # 首先对每一层进行二维卷积
        for i in range(r):
            c2d_result[i, :, :] = self.__quick_convolute_2d(data[i, :, :])
        # 对二维卷积后的矩阵换一个维度再次进行卷积，得到三维卷积结果
        c3d_result = np.zeros((r, c, h), dtype=np.int)
        for j in range(c):
            c3d_result[:, j, :] = self.__convolution_x(c2d_result[:, j, :], self.kernel_radius[2])
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
        max_val = int(math.pow(2, self.depth_bits))
        for i in range(max_val):
            for j in range(max_val):
                for k in range(max_val):
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
        max_val = int(math.pow(2, self.depth_bits))
        color_matrix = np.zeros((max_val, max_val, max_val))
        dealed = {}
        rows, cols, channels = img.shape
        for i in range(rows):
            for j in range(cols):
                b, g, r = img[i, j]
                color_matrix[b, g, r] = 1
                dealed[(b, g, r)] = -1
        return color_matrix, dealed

    def cluster_to_single_image(self, img, clusters):
        '''在一张图像中显示分割结果'''
        rows, cols, channels = img.shape
        segmented_image = np.ones(img.shape, np.uint8) * 155
        for r in range(rows):
            for c in range(cols):
                color = tuple(img[r, c])
                label = clusters[color]
                segmented_image[r, c] = color_list[label]
        return segmented_image


def my_show(win_name, src):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, src)

if __name__ == '__main__':
    '''
    功能测试，入口函数
    '''
    start = time.time()
    # img = cv2.imread(r'E:\MachineLearning\ImageSegmentation_Cluster\capsule_images\origin\capsule_1.bmp')
    img = cv2.imread(r'E:\MachineLearning\ImageSegmentation_Cluster\capsule_images\xdpi\capsule_2.bmp')
    conDB = ConDBSCAN(np.ones((3, 7, 9)), 60)
    data, dealed = conDB.image_to_3DMatrix(img)
    print('数据集大小： ' + str(len(dealed)))
    clusters, labels = conDB.fit(data, dealed)
    end = time.time()
    print('程序运行时间：%f s' % (end - start))

    print('分类簇个数：%d' % len(labels))

    seg_img = conDB.cluster_to_single_image(img, clusters)
    my_show('segementation', seg_img)
    cv2.waitKey(0)
