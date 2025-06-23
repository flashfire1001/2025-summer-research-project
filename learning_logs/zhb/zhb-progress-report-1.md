# 做了什么

1. 完成了第1-4章内flow-matching的数学原理学习收尾 ，基本完全理解mit note 1-4章里有关flow-matching部分的论述（score matching和diffusion之类的没有看，跳过了）。顺便学了continuity equation和一些多元微积分的知识。
2. 了解了MLP背后的计算原理，理解了前向/反向传播的机制（有向无环图），并实现了MLP分类MNIST数据集的一个小[lab](https://github.com/hanbinzheng/generating-model/tree/main/python-baiscs/pytorch/NN). 然而，我还是证明不出来为什么MLP可以拟合任何函数，我只能感性的模糊的理解。还有我发现，我做完了lab，只能保证自己能看懂这个小lab，但是不给任何参考的情况下自己敲我完全敲不出来。
3. 理解并且跑通mit的lab1和lab2的内容（没有学他的可视化，lab2只自己敲了flow-matching有关的，score-matching有关的我没有敲，只是跑了一遍）
4. 试图简化lab1和lab2做一个flow-matching的[toy image generator](https://github.com/hanbinzheng/generating-model/tree/main/failed_proj)，即从二维的标准高斯噪声生成moons/checkerboard/circles，但是失败了。这个mlp的loss不随着时间收敛，根本没用。

# 接下来一周要做什么

1. 完成上面没做成功的toy image generator
2. 学神经网络。我的toy image generator失败的原因是神经网络训练失败。了解一下各种神经网络，还有神经网络训练时候的各种神奇的方法，比如什么正则化，什么dropout，什么normalize，什么 weight_decay，什么gradient clipping，还有什么latent space。这些是我问gpt让gpt解决问题我toy image generator问题的时候gpt回答的办法，但是我一个都不知道。
3. 学画图，学可视化（mit lab里的可视化代码看不懂，而且我确实不会画图，学一下）
4. 如果还有时间（大概率是没时间了），学mit note的第5章还有看lab3