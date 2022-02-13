本项目用来以最简单的demo的形式展示神经网络的应用过程，用来串联其中需要用到的技术

- 1.数据准备
- 2.模型训练
- 3.模型部署


依赖环境：
opencv
pytorch

执行顺序：
- 1.执行DataCollection中的mnist2image.py文件，会转换原始数据为训练测试要用到的jpg和txt
- 2.执行Training中的train.py文件，完成模型训练，并保存当前模型和训练过程
- 3.执行Development中的use_demo.py可以转换你自己手写的数字为模型推理结果

参考文献：
- 评价指标：https://juejin.cn/post/7033753474089091086
- pytorch模型转onnx模型：https://blog.csdn.net/infinite_jason/article/details/117660030