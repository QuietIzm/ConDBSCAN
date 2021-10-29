# # ConDBSCAN

### 介绍

面向图像分割领域，引入卷积思想优化了**DBSCAN**算法，提出了卷积优化密度聚类算法**ConDBSCAN**，将**DBSCAN**算法的时间复杂度从**O(*n*^2^)** 降低到 **O(*n*)**，提高了密度聚类算法在图像处理领域的可用性。

本函数库中包含**ConDBSCAN**算法的两种实现版本：C++/Python。

### 环境依赖

C++ 和 Python 的实现版本均需要 OpenCV 的支持，在使用本算法库的时候请先配置好 OpenCV。



### 用法

**ConDBSCAN by C++**：

> 将库中C++目录下面的 **ConDBSCAN.cpp** 和 **ConDBSCAN.h **文件拷贝到自己的工程目录下，进行如下操作：
>
> - 添加如下引用 `#include "ConDBSCAN.h"`；
> - 实例化`ConDBSCAN`类，调用对应的接口函数。

**ConDBSCAN by Python:**

> 将库中Python目录下面的  **dbscan_by_convolution.py** 文件拷贝到自己的工程目录下，进行如下操作：
>
> - 添加如下引用 `from dbscan_by_convolution import ConDBSCAN`；
> - 实例化`ConDBSCAN`类，调用对应的接口函数。

### 函数接口

以下为**ConDBSCAN**中提供调用的函数接口：

**ConDBSCAN by C++:**

|          函数名          | 输入参数                                                     |          返回值          |                             描述                             |
| :----------------------: | :----------------------------------------------------------- | :----------------------: | :----------------------------------------------------------: |
|      **ConDBSCAN**       | ***kernel***: OpenCV中 Point3i(等价于vector<int, int, int>)类型，表示卷积核<br>***minpts***:  DBSCAN中的阈值参数 |            /             |                           构造函数                           |
|   **ImageTo3DMatrix**    | ***img***: 输入图像<br>***colorMatrix***:图像颜色值映射为的三维矩阵<br>***dealed***:保存图像中的颜色信息 |            /             |          将图像**img**映射为三维矩阵**colorMatrix**          |
|         **Fit**          | ***colorMatrix***:输入颜色矩阵<br>***dealed***:保存聚类结果<br>***labels***:保存聚类簇标签 |            /             |          对输入的是三维矩阵**colorMatrix**进行聚类           |
|  **ClusterToMutiImage**  | ***labels***:聚类簇标签<br/>***cluster***:聚类结果<br/>***img***:输入图像 | 返回聚类簇对应的图像集合 |   将聚类结果转换成图像分割结果，每幅黑白图像代表一个聚类簇   |
| **ClusterToSingleImage** | ***labels***:聚类簇标签<br/>***cluster***:聚类结果<br/>***img***:输入图像 |     返回图像分割结果     | 将聚类结果转换成图像分割结果，以一幅彩色图像表示，图像中不同的颜色表示不同的聚类簇 |
|    **CalSilhouette**     | ***results***:聚类结果<br/>***origin***:原始数据，需转换为哈希表<br/>***num***:聚类簇个数 |    返回计算出的SI系数    |       计算聚类结果的SI系数，用来描述聚类效果(实验方法)       |

**ConDBSCAN by Python:**

|           函数名            | 输入参数                                                     |              返回值              |                             描述                             |
| :-------------------------: | ------------------------------------------------------------ | :------------------------------: | :----------------------------------------------------------: |
|        **ConDBSCAN**        | ***kernel***:定义卷积核，numpy类型一维数组<br>***minpts***:dbscan 阈值参数 |                /                 |                           构造函数                           |
|           **fit**           | ***data***:图像映射的三维矩阵<br>***dealed***:保存聚类结果   |       聚类结果和聚类簇标签       |                  对输入的是三维矩阵进行聚类                  |
|    **image_to_3DMatrix**    | ***img***:输入图像                                           | 映射的三维矩阵和保存颜色值的字典 |                     将图像映射为三维矩阵                     |
| **cluster_to_single_image** | ***img***:输入图像<br>***clusters***:聚类结果                |         返回图像分割结果         | 将聚类结果转换成图像分割结果，以一幅彩色图像表示，图像中不同的颜色表示不同的聚类簇 |
|         **my_show**         | ***win_name***:窗口名<br/>***src***:带显示图像               |                /                 | 显示图像，未在`ConDBSCAN`类中定义，需单独引用`from dbscan_by_convolution import my_show` |

### License

***Copyright@QuietIzm***
