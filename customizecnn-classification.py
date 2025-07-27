import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models, optimizers, regularizers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
import os

# 定义预处理函数

# 归一化图像到 [0,1] 范围
def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label
    
# 数据增强函数
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    return image, label

# 设置seed
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # GPU相关设置
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
set_seed(10)

# 加载数据集
(train_ds, test_ds), ds_info = tfds.load(
    "eurosat/rgb",
    split=["train[:80%]", "train[80%:]"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# 修正数据管道
def prepare_dataset(ds, batch_size=32, use_augment=False):  
    ds = ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    if use_augment:  # 使用 use_augment 判断是否启用增强
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)  
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# 创建数据加载器
train_ds = prepare_dataset(train_ds, batch_size=64, use_augment=True)  
test_ds = prepare_dataset(test_ds, batch_size=64)# 获取类别名称和数量
class_names = ds_info.features['label'].names
num_classes = len(class_names)
print("Class Counts:", num_classes)
print("Class Names:", class_names)

# 创建一个字典来存储每个类别的第一个样本
class_samples = {name: None for name in class_names}

# 遍历数据集直到每个类别都找到一个样本
for images, labels in train_ds.unbatch():
    label_name = class_names[labels.numpy()]
    if class_samples[label_name] is None:
        class_samples[label_name] = (images.numpy(), label_name)
    # 检查是否所有类别都已找到样本
    if all(sample is not None for sample in class_samples.values()):
        break

# 绘制每个类别的样本
cols = 5
rows = (num_classes + cols - 1) // cols
plt.figure(figsize=(15, 3 * rows))

for idx, (class_name, (image, _)) in enumerate(class_samples.items()):
    plt.subplot(rows, cols, idx + 1)
    plt.imshow(image)
    plt.title(class_name, fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()def build_cnn_model(input_shape=(64, 64, 3), num_classes=10):
    model = models.Sequential([
        # 卷积块1 + 正则化
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, 
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),  # Dropout层

        # 卷积块2
        layers.Conv2D(128, (3, 3), activation='relu', 
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # 卷积块3
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        
        # 全连接层
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),  # 高Dropout防止过拟合
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
    
model = build_cnn_model()
# 输出模型信息
model.summary()

# 回调函数
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5", 
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2,      # 学习率乘以0.2
        patience=3,      # 连续3次验证损失不下降时触发
        min_lr=1e-7     # 最小学习率下限
    )
]

# 编译模型
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),  # 初始学习率可稍大
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)# 训练
epochs = 100 # 存在早停策略，可设置较大值
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs,
    callbacks=callbacks
)

# 评估测试集
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")
# 获取实际训练轮数
actual_epochs = len(history.history['accuracy'])
print("Setting epochs:",epochs)
print("Actually epochs:",actual_epochs)

# 绘制评测曲线（accuracy+loss）
plt.figure(figsize=(15, 5))

# Accuracy曲线
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.xticks(range(0, actual_epochs), range(1, actual_epochs+1))  # 使用actual_epochs
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Loss曲线
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.xticks(range(0, actual_epochs), range(1, actual_epochs+1))  # 使用actual_epochs
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
def evaluate_per_class_accuracy(model, test_ds, class_names):
    # 配置数据集线程参数
    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_threading.private_threadpool_size = 4
    test_ds = test_ds.with_options(options)
    
    # 批量预测
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_pred = model.predict(test_ds, verbose=0).argmax(axis=1)
    
    # 打印分类报告
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # 计算每个类别的准确率
    cm = confusion_matrix(y_true, y_pred)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name:25s}: {class_accuracy[i]:.4f}")
    
    # 可视化混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()
evaluate_per_class_accuracy(model, test_ds, class_names)
# 选择一批测试数据（6张图）并展示
def plot_predictions(model, test_dataset, class_names, num_images=6):
    plt.figure(figsize=(15, 10))
    shuffled_ds = test_dataset.shuffle(buffer_size=100)
    for images, labels in shuffled_ds.take(1):  # 取一个batch
        # 预测
        predictions = model.predict(images)
        pred_labels = tf.argmax(predictions, axis=1).numpy()
        
        # 绘制图像和预测结果
        for i in range(min(num_images, len(images))):
            plt.subplot(2, 3, i+1)
            img = images[i].numpy()
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])  # 反归一化（ImageNet均值）
            img = np.clip(img, 0, 1)
            plt.imshow(img)
            plt.title(f"True: {class_names[labels[i]]}\nPred: {class_names[pred_labels[i]]}")
            plt.axis('off')
        plt.show()
        break 

# 展示预测
plot_predictions(model, test_ds, class_names)