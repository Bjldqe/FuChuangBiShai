import tensorflow as tf
import os
from keras.layers import Flatten, Dense, Dropout, Input
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical # 用于one-hot编码
from keras import utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from tensorflow import keras
import efficientnet.tfkeras as efn
epochs=300
height=128 
width=128                                             
class_num=7
batch_size=32
#训练集
X_train=np.load('../tmp/data/X_train.npy')
Y_train=np.load('../tmp/data/Y_train.npy')

base_model = efn.EfficientNetB0(input_shape=[128,128,3], include_top=False,weights='imagenet')
base_model.trainable =  True
# 冻结前面的卷积层，训练最后10层
for layers in base_model.layers[:-10]:
    layers.trainable =False
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(class_num,activation='softmax')
])
#模型可视化1，使用model.summary()方法
model.summary()
model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.001,momentum=0.9, decay=1E-5),#优化器
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),#损失率计算
              metrics=["accuracy"])#监视器，打印准确率
 # 创建保存模型的文件夹
if not os.path.exists("../tmp/save_weights"):
    os.makedirs("../tmp/save_weights")
#保存模型，回调函数
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='../tmp/save_weights/EfficientNetB0.h5',#保存模型位置和命名
                                                save_best_only=True,#保存最佳的参数
                                                save_weights_only=False,#为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
                                                monitor='val_loss')]#监视验证集损失函数，判断最佳参数
# 开始训练
print("开始训练网络···")
from sklearn.model_selection import KFold
KF = KFold(n_splits =10)  #建立4折交叉验证方法 查一下KFold函数的参数
for train1_index, val1_index in KF.split(X_train):
    x_train1, y_train1 = X_train[train1_index], Y_train[train1_index]
    x_val1, y_val1 = X_train[val1_index], Y_train[val1_index]
history=model.fit(  x_train1,
                    y_train1,  #训练集
                    shuffle=True,#打乱数据
                    steps_per_epoch=len(X_train) // batch_size,  #迭代频率
                    epochs=epochs,  #迭代次数
                    validation_data=(x_val1,y_val1),
                    callbacks=callbacks)
