# How our summer research project will be carried out?

xjm_编辑

>   Stage2:完成一个基于流模型的 Image Restoration (图片复原) 改善这个模型,若有可能,写一篇小论文

基本上:

1.   学习优化器,调参数的知识. 比如scheduler
2.   思考怎么优化Unet和t-embedding (哪里可以改进)
3.   思考解决不同模样的y带来的不同. 我的方案是: 利用一个基础的adapter ( = conv+BN+Silu+conv+BN +Silu)增加channel是,之后由于是一同样的channel可以加到全部数据中去.或者简单粗暴直接加上去.

分工:

1.   训练出模型(最好flow_matching+mean_flow都要) 因为非常可能失败(那就啥都没了)  
2.   做各种测试(见share_resources中的文章)
3.   写论文(每个人写一部分, 内容参照google.pdf 简练清晰)
4.   快点完成, 我们还要花很多时间训练(一周就跑出一个模型…是吗?)

