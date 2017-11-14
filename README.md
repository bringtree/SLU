```
.
├── README.md
├── __pycache__
│   └── util.cpython-36.pyc
├── data 数据集
│   ├── aaa.iob 用来判断代码有没有写错
│   ├── atis.test.w-intent.iob 测试集(main.py用的数据集)
│   ├── atis.train.w-intent.iob 训练集(test.py用的数据集)
│   └── bbb.iob 用来判断代码有没有写错
├── false.txt (test.py 运行测试集合的时候 预测的结果)
├── main.py (训练代码)
├── ready.txt (atis.test.w-intent.iob 中提取出来的真实结果)
├── stopwords (用来去掉语气词 等等)
│   └── english
├── test.py
├── train_recorder.js (随机森林中各个超参跑出来的性能值)
├── util.py (一些预处理操作)
└── 决策树性能图
    ├── max_depth.png
    ├── max_leaf_nodes.png
    ├── min_samples_leaf.png
    ├── min_samples_split.png
    ├── min_weight_fraction_leaf.png
    └── n_estimators.png

```

####
早期遇到一个坑，自己把意图用one-hot 编码进行预处理后再作输出值，用决策树的时候会出现异常(结果不准确)。
详见[https://github.com/scikit-learn/scikit-learn/issues/9988]