# EuroSAT RGB 卫星图像分类 - 自定义CNN实现
# 版本
**原版本共有约44w参数，输出模型约6MB，验证集准确率94%  
调整后的低参数版本共有约9w，输出模型约1MB，验证集准确率93.5%**
## 项目概述
本Python脚本实现了一个自定义卷积神经网络(CNN)，用于对EuroSAT RGB数据集中的卫星图像进行分类。该数据集包含10种不同的地表覆盖类型，本模型通过数据增强、正则化技术和高级回调函数优化训练过程，实现高精度分类。

## 主要功能
- **数据预处理**：图像归一化到[0,1]范围
- **数据增强**：随机左右翻转和亮度调整
- **自定义CNN架构**：
  - 3个卷积块（64→128→256滤波器）
  - 批归一化(Batch Normalization)
  - L2正则化和Dropout防止过拟合
  - 全局平均池化层
- **高级训练技术**：
  - 早停(EarlyStopping)策略
  - 学习率动态调整(ReduceLROnPlateau)
  - 模型检查点保存
- **全面评估**：
  - 整体准确率和损失曲线
  - 每类准确率分析
  - 混淆矩阵可视化
  - 样本预测展示

## 环境要求
```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy tensorflow-datasets
```

## 使用说明
1. 在jupyter notebook环境下打开customizecnn-classification.ipynb
2. 执行全部运行
3. 程序将自动执行以下流程：
   - 下载并预处理EuroSAT RGB数据集
   - 构建并训练自定义CNN模型
   - 评估模型性能
   - 生成多种可视化结果


## 组件说明
- **数据管道**：使用`tf.data`高效加载和处理数据
- **正则化技术**：
  - Dropout层(20-50%)
  - L2权重正则化(λ=1e-4)
- **回调函数**：
  - 早停(连续5轮验证损失无改善则停止)
  - 学习率动态调整(损失停滞3轮后降低为20%)
  - 最佳模型保存
- **评估指标**：
  - 整体准确率
  - 每类准确率
  - 分类报告(精确率、召回率、F1分数)
  - 混淆矩阵

## 数据集信息
EuroSAT RGB数据集包含27,000张64×64像素的卫星图像，分为10个类别：
1. AnnualCrop (一年生作物)
2. Forest (森林)
3. HerbaceousVegetation (草本植被)
4. Highway (高速公路)
5. Industrial (工业区)
6. Pasture (牧场)
7. PermanentCrop (多年生作物)
8. Residential (住宅区)
9. River (河流)
10. SeaLake (海洋/湖泊)
<img width="1119" height="596" alt="image" src="https://github.com/user-attachments/assets/6f369d84-efe4-4bb0-8d8c-1bf97f41a970" />


数据集按80%训练集和20%测试集划分

## 自定义选项
1. **模型架构**：
   - 修改`build_cnn_model()`中的层结构
   - 调整正则化参数
2. **训练参数**：
   - 批次大小：修改`prepare_dataset()`中的`batch_size`
   - 学习率：调整`optimizers.Adam()`中的初始学习率
   - 早停耐心值：修改`EarlyStopping`中的`patience`
3. **数据增强**：
   - 在`augment_image()`中添加新增强方法
   - 调整现有增强参数

## 性能表现
经过训练该模型在EuroSAT测试集上能达到：
- **整体准确率**：>90%
- **详细每类准确率**：输出在终端和可视化中展示
- **训练效率**：通过早停机制自动确定最优训练轮数
<img width="1103" height="468" alt="image" src="https://github.com/user-attachments/assets/b675a794-fce5-456f-9973-e63073eff4ed" />
<img width="566" height="716" alt="image" src="https://github.com/user-attachments/assets/aa26de54-249b-4cf6-825b-030ce2871b32" />



> 注：实际性能可能因运行环境略有差异，随机种子已固定为10以保证可复现性
