# Constrained Mean Shift Clustering

This is the official implementation of [*Constrained Mean Shift Clustering*](http://www.tnt.uni-hannover.de/papers/data/1553/CMS.pdf).

![CMS Animation on Moons data set](ReadmeMoons.gif)

Constrained Mean Shift (CMS) is a novel approach for mean shift clustering under sparse supervision 
using cannot-link constraints. The constraints provide a guidance in constrained clustering 
indicating that the respective pair should not be assigned to the same cluster. 
Our method introduces a density-based integration of the constraints to generate individual 
distributions of the sampling points per cluster. We also alleviate the (in general very sensitive) 
mean shift bandwidth parameter by proposing an adaptive bandwidth adjustment which is especially 
useful for clustering imbalanced data sets.

约束均值偏移 (CMS) 是一种使用无法链接约束在稀疏监督下进行均值偏移聚类的新颖方法。这些约束提供了约束聚类的指导，指示不应将相应的对分配到同一聚类。我们的方法引入了基于密度的约束集成，以生成每个簇的采样点的单独分布。我们还通过提出自适应带宽调整来减轻（通常非常敏感）均值漂移带宽参数，这对于聚类不平衡数据集特别有用

## Brief Usage

Given some data points and some binary cannot link constraints:

给定一些数据点和一些二元不能链接约束：

```python
from sklearn.datasets import make_moons

# Generate moons data set
x, y = make_moons(shuffle=False)
# Create one cannot-link constraint from center of one moon to another
cl = [[25, 75]]
```

CMS can be invoked similar to sklearn cluster methods:

CMS可以像sklearn集群方法一样调用：

```python
from CMS import CMS, AutoLinearPolicy

# Create bandwidth policy as used in our experiments
pol = AutoLinearPolicy(x, 100)
# Use nonblurring mean shift (do not move sampling points)
cms = CMS(pol, max_iterations=100, blurring=False)
cms.fit(x, cl)
```

The `cms` object now contains the following members:

`cms` 对象现在包含以下成员：

Member | Description
--- | ---
`labels_` | Final cluster labels 最终群集标签
`modes_` | Final position of the cluster centers/modes 集群中心/模式的最终位置
`bandwidth_history_` | Bandwidths used per iteration 每次迭代使用的带宽
`mode_history_` | Cluster centers/modes per iteration 每次迭代的集群中心/模式
`kernel_history_` | Kernel weights per iteration 每次迭代的内核权重
`block_history_` | Attraction reduction per iteration 每次迭代的吸引力减少

To visualize the results, we provide a convenient Matplotlib routine:

为了可视化结果，我们提供了一个方便的 Matplotlib 例程：
```python
from CMS.Plotting import plot_clustering
import matplotlib.pyplot as plt

plot_clustering(x, cms.labels_, cms.modes_, cl=cl)

plt.show()
```

You can run [example_moons.py](example_moons.py) to try it yourself. You may also try adjusting the parameters of CMS:

您可以运行 [example_moons.py](example_moons.py) 自行尝试。您也可以尝试调整CMS的参数：

Parameter | Description
--- | ---
``h`` | Set the bandwidth either to a scalar float value, or a callable ``f(int) -> float`` returning the bandwidth for the given iteration 将带宽设置为标量浮点值，或可调用的“f（int） -> float”，返回给定迭代的带宽
``max_iterations`` | Maximum number of iterations 最大迭代次数
``blurring`` | If ``True`` use blurring mean shift, i.e. the sampling points are updated with the cluster centers after each iteration, thus blurring them in the process. If ``False`` use nonblurring mean shift, where sampling points remain stationary. 如果为“True”，使用模糊均值偏移，即采样点在每次迭代后都会使用聚类中心进行更新，从而在此过程中模糊它们。如果为“False”，则使用非模糊均值偏移，其中采样点保持静止
``kernel`` | If ``'ball'``, use a ball kernel, otherwise expects a float in range [0, 1) to use as truncation of a truncated Gaussian kernel. Thus setting ``kernel=0.`` uses a regular Gaussian kernel. 如果是“ball“”，使用球核，否则期望 [0， 1] 范围内的浮点数用作截断的高斯核的截断。因此，设置“kernel=0.”使用常规高斯核
``c_scale`` | The constraint scaling parameter that determines the spatial influence of constraints. For lower values, constraints have less reducing influence on far attractions. 确定约束的空间影响的约束缩放参数。对于较低的值，约束对远处景点的递减影响较小
``label_merge_k`` | This implementation of CMS uses connected components to determine the final cluster labels from the final cluster centers. Specifies the minimum closeness in terms of kernel value to merge two cluster centers into one cluster. CMS 的此实现使用连接的组件来确定来自最终集群中心的最终集群标签。指定将两个集群中心合并为一个集群的最小接近度（以内核值为单位）
``label_merge_b`` | Specifies the lowest weight reduction through constraints below which two cluster centers are never merged. Set to ``0.`` to disable. 通过约束指定最低的权重降低，低于该约束值时，两个聚类中心永远不会合并。设置为“0.”以禁用
``use_cuda`` | If ``True``, use the CUDA Toolkit to accelerate some calculations. You must have the CUDA Toolkit installed. Please consult the official CUDA documentation on how to install CUDA for your specific system. 如果为“True”，请使用 CUDA 工具包加速某些计算。您必须安装 CUDA 工具包。请参阅CUDA官方文档，了解如何为您的特定系统安装CUDA

### Local

To run the experiments, you can create a Conda environment with the required dependencies by running the following commands. This will install most dependencies with the exact version used during out experiments.

要运行实验，您可以通过运行以下命令来创建具有所需依赖项的 Conda 环境。这将使用实验期间使用的确切版本安装大多数依赖项。
```shell
conda create --name cms python=3.8
conda activate cms
pip install -r requirements.txt
```
## Experiments

### Synthetic Data

First, you must download the used synthetic data sets by running `./download_synth.sh`. To evaluate performance of CMS on the synthetic data sets, run `python cluster_synth.py --data <DATA>`, where `<DATA>` is one of `moons`, `jain`, `s4`, or `aggregation`, e.g., 

首先，您必须通过运行“./download_synth.sh”下载使用的合成数据集。要评估 CMS 在合成数据集上的性能，请运行“python cluster_synth.py --data <DATA>”，其中“<DATA>”是“moons”、“jain”、“s4”或“aggregation”之一，例如，
```shell
python cluster_synth.py --data aggregation
```

