# tensorflow-ocr

### 中文汉字印刷体识别
修改自这个[仓库](https://github.com/soloice/Chinese-Character-Recognition)，将训练与预测分隔出来，便于作为接口提供识别服务

### config.py
定义了识别时的图片大小，训练字符集，模型位置等

### preprocess.py
用于对输入图片进行预处理

### model.py
模型定义

### train.py
模型训练模块

### predict.py
模型预测模块
