import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix,classification_report #导入混淆矩阵函数 
import matplotlib
from keras.applications import resnet50,mobilenet_v2
from keras.models import Model
from tensorflow import keras
import efficientnet.tfkeras as efn
#输入图片的大小,需要与存储的图片大小一致
height=128
width=128
#输入模型类别个数
class_num = 7
#测试集和测试集标签所在位置
test_X='../tmp/data/X_test.npy'
test_Y='../tmp/data/Y_test(label).npy'

base_model = efn.EfficientNetB0(input_shape=[height,width,3], include_top=False,weights='imagenet')
base_model.trainable =  True
# 冻结前面的卷积层，训练最后10层
for layers in base_model.layers[:-10]:
    layers.trainable =False
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(class_num,activation='softmax')
])
model.load_weights('../tmp/save_weights/EfficientNetB0.h5')

# 取消冻结模型的顶层
base_model.trainable = True
fine = 129
for layer in base_model.layers[:fine]:
    layer.trainable = False
model = keras.Sequential([
    base_model,
    #keras.layers.Dropout(0.25),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(class_num,activation='softmax')
])
#导入模型权重文件
model.load_weights('../tmp/save_weights/EfficientNetB0_wt.h5')

#导入测试数据
test_X=np.load(test_X)
#导入测试类别标签
test_Y_True=np.load(test_Y)

#对测试集进行预测
pre_result = model.predict_classes(test_X)
#预测结果-真实结果
Z=pre_result-test_Y_True
#预测正确的结果/所有的预测值
test_acc=len(Z[Z==0])/len(Z)
print('预测准确率为：',test_acc)

cm = confusion_matrix(test_Y_True, pre_result) #混淆矩阵   
plt.matshow(cm, cmap=plt.cm.Greens) #画混淆矩阵图，配色风格使用cm.Greens，更多风格请参考官网。 
plt.colorbar() #颜色标签  
# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
for x in range(len(cm)): #数据标签  
    for y in range(len(cm)):  
        plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')  
        plt.ylabel('True label') #坐标轴标签  
        plt.xlabel('Predicted label') #坐标轴标签  
#显示混淆矩阵可视化结果
plt.savefig('../tmp/混淆矩阵.jpg',dpi=3090)
plt.show()
print(classification_report(test_Y_True,pre_result))


