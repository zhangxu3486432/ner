# 项目说明文件和性能说明文件

## 数据集

ACL 2018 [Chinese NER using Lattice LSTM](https://link.zhihu.com/?target=https%3A//github.com/jiesutd/LatticeLSTM) 中从新浪财经收集的简历数据

## 数据格式

CoNLL 格式（BIOES标注模式），每一行包含字符和标签，中间用空格分隔。

```
美	B-LOC
国	E-LOC
的	O
华	B-PER
莱	I-PER
士	E-PER

我	O
跟	O
他	O
谈	O
笑	O
风	O
生	O 
```

## 模型

**在这个项目中，使用的是多层双向 LSTM 模型。**下面我们将对模型进行简单的介绍。

### LSTM

LSTM 中引入了3个门，即输入门（input gate）、遗忘门（forget gate）和输出门（output gate），以及与隐藏状态形状相同的记忆细胞（某些文献把记忆细胞当成一种特殊的隐藏状态），从而记录额外的信息。

![](https://cdn.jsdelivr.net/gh/zhangxu3486432/zhangxu3486432.github.io/static/images/20200614123048.png)

### Stacked LSTM

在深度学习应用里，我们通常会用到含有多个隐藏层的循环神经网络，也称作深度循环神经网络。每个隐藏状态不断传递至当前层的下一时间步和当前时间步的下一层。

![](https://cdn.jsdelivr.net/gh/zhangxu3486432/zhangxu3486432.github.io/static/images/20200614124032.png)

### BiLSTM

之前介绍的循环神经网络模型都是假设当前时间步是由前面的较早时间步的序列决定的，因此它们都将信息通过隐藏状态从前往后传递。有时候，当前时间步也可能由后面时间步决定。例如，当我们写下一个句子时，可能会根据句子后面的词来修改句子前面的用词。双向循环神经网络通过增加从后往前传递信息的隐藏层来更灵活地处理这类信息。

![](https://cdn.jsdelivr.net/gh/zhangxu3486432/zhangxu3486432.github.io/static/images/20200614123756.png)

## 安装依赖

```bash
conda install scikit-learn, numpy
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

## 运行

```bash
python train.py

# or

python train.py --epochs=30 --batch_size=64 --weight_decay=1e-4 --num_layers=2 --hidden=128 --embedding=128 --dropout=0.5 --lr=0.001
```

## 性能

在测试集上的结果

```
Test, Loss: 0.2211, Acc: 0.9513
Test, Classification report:
               precision    recall  f1-score   support

     M-TITLE       0.93      0.89      0.91      1922
      B-RACE       1.00      1.00      1.00        14
       B-ORG       0.93      0.97      0.95       553
       M-ORG       0.94      0.97      0.96      4325
       E-ORG       0.91      0.92      0.91       553
      E-CONT       1.00      1.00      1.00        28
      M-RACE       0.00      0.00      0.00         0
       M-LOC       0.68      0.81      0.74        21
      S-RACE       0.00      0.00      0.00         0
       E-LOC       0.71      0.83      0.77         6
      B-CONT       1.00      1.00      1.00        28
     E-TITLE       0.98      0.98      0.98       772
           O       0.98      0.96      0.97      5190
       M-EDU       0.95      0.92      0.93       179
      S-NAME       0.00      0.00      0.00         0
       B-EDU       0.98      0.95      0.96       112
       E-EDU       0.98      0.95      0.96       112
      E-NAME       1.00      0.98      0.99       112

   micro avg       0.95      0.95      0.95     13927
   macro avg       0.78      0.78      0.78     13927
weighted avg       0.95      0.95      0.95     13927
```
