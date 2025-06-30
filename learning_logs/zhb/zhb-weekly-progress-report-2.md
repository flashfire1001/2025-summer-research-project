本周完成以下内容：

1. 修改之前失败的地方，发现失败的原因：
   - 算 loss 的时候 xt 应该从 p_t (x) 采样，但是我代码里错误的从 p_0 (x) 采样。这个是最主要的失败结果
   - 学习的图像是黑白的，也就是0和255，不是0-255，而且没有归一化
   
     这两个原因，特别是第一个，是没能做成功，loss不收敛的主要原因。

     在修改好这两个错误之后，重构了一下。

     重构之后发现，loss是可以在震荡中收敛了，但是生成效果还是很不好。经过各种调试，比如改变mlp架构，调lr，调整epochs，效果还是很不好。只是勉强能看出moons/checkerboard这些东西的形状，但是高斯噪声还是很明显。

     我觉得主要原因是，mlp是4097(64 * 64 + 1)的输入，这个时间维度t几乎没什么影响了，mlp学不出来。

     https://github.com/hanbinzheng/generating-model/tree/main/small_project/toy_examples

2. 学习CNN和UNet的内容
   - 了解了CNN的基本内容，比如conv，stride，padding，pooling等内容，了解CNN是“extract abstract feature“ （体现在feature量上升，但是矩阵范围变小）
   - 了解反卷积，搞明白了ConvTranspose的基本内容
   - 了解UNet架构，encoder + midcoder + decoder，还有如何把原来细节通过residual connection从encoder传递到decoder。decoder从某种意义上，就是融合原来的细节然后加强学到的特征

3. 完成lab3，按照lab3的框架用unet生成了mnist
   发现unet里面对于时间t和guidance y的融合是mlp不能比的。他把t先用傅立叶给编成向量，然后用一个小的mlp把这个向量变换维度，使他可以和图像的tensor广播。对y也进行了一个embedding，然后和t一样mlp变换维度。
   在embedding完y和t之后，他把y和t直接加到要处理的图像x中，然后利用CNN来处理提取信息

   虽然不知道为什么这个方法能奏效，但是感觉这种办法显然比mlp的“把t当作一个新的维度直接放到输入”有效


正在做的内容：
根据lab3的框架，尝试cifar10的生成
