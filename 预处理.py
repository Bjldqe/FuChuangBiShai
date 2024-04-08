import numpy as np
import cv2
import os
import pandas as pd
#图片路径
file_path = '../data/images/'
data_label=pd.read_csv('../data/rock_label.csv',encoding='gbk')
img_width=[]
img_hight=[]
dimen=[]
for i in os.listdir(file_path):
    image=cv2.imread(file_path+i)
    #查看图片尺寸
    h,w,d = image.shape
    #依次保存所有图像高
    img_hight.append(h)# 高
    img_width.append(w)#宽
    dimen.append(d)
#图片像素大小探索数据
img_hight = pd.DataFrame(img_hight, columns=list({'hight'}))
img_width = pd.DataFrame(img_width, columns=list({'width'}))#保存为数据框形式并设置列索引
img_dimen = pd.DataFrame(dimen, columns=list({'dimension'}))
Imgdata = pd.concat([img_hight,img_width, img_dimen],axis=1)#合并数据
#对数据进行描述性分析,探索数据
Imgdata.describe() 

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'         #设置字体为SimHei
data_class=data_label['样本类别'].unique()
for d in data_class:
    dt=data_label['样本类别'].value_counts(d)
# 2.绘制饼图
plt.pie(dt, labels=data_class,
        autopct='%1.2f%%')  # 绘制饼图,百分比保留小数点后两位
plt.title('各岩石样本类别百分比饼图')
plt.savefig('../tmp/饼图.jpg',dpi=3090)



import os
import pandas as pd
#导入标签值
data_label=pd.read_csv('../data/rock_label.csv',encoding='gbk')
file_path = '../data/images/' #原始数据文件夹
#创建类别文件夹
save='../tmp/class_data/'
for i in list(data_label['样本类别'].unique()):
    # 生成相应文件夹
    lei_path = save+str(i)
    if not os.path.exists(lei_path):
        os.mkdir(lei_path)
import cv2
import numpy as np
import tensorflow as tf
# 目标提取
def getMB(image):
     image1=image[620:1650,645:1650]
     return image1
#随机裁剪图像
data_class=data_label['样本类别'].unique()
for cla in list(data_class):
    label=data_label.loc[data_label['样本类别']==cla,:]
    hsm1=label['样本编号'].tolist()
    #根据样本编号选取选取相应类别图片随机切割保存到相应文件夹中
    for num in hsm1:
        if num<322: 
            for i in range(15):
                image = cv2.imdecode(np.fromfile(file_path+str(num)+'.bmp', dtype=np.uint8),-1)
                #将图片进行随机裁剪
                size=int(512)
                crop_img = tf.image.random_crop(image,[size,size,3])
                crop_img=np.array(crop_img)
                name=cla+'-'+str(num)+'-'+str(i)+'.jpg'
                cv2.imencode('.jpg',crop_img)[1].tofile(save+cla+'/'+name)
        else:
            for j in range(15):
                image1 = cv2.imdecode(np.fromfile(file_path+str(num)+'.jpg', dtype=np.uint8),-1)
                image1=getMB(image1)
                #随机裁剪
                size1=int(512)
                crop_img1 = tf.image.random_crop(image1,[size1,size1,3])
                crop_img1=np.array(crop_img1)
                name1=cla+'-'+str(num)+'-'+str(j)+'.jpg'
                cv2.imencode('.jpg',crop_img1)[1].tofile(save+cla+'/'+name1)

#获取数据文件夹下的类别名称存为一个列表形式
data_class = [cla for cla in os.listdir(save) if os.path.isdir(os.path.join(save, cla))] 
num_class=[] 
for cla in data_class:
    #获取类别图像路径
    adrss = os.path.join(save, cla)
    #读取路径下的所有图片
    images = os.listdir(adrss)
    #查看每种类别文件夹的图像数量
    num = len(images)
    # 保存类别及对应数量
    num_class.append([cla,num])
    #输出类别及对应数量
    print('{}:{}张'.format(cla,num))

from pre_model import horizon_flip,vertical_flip,rotate,horandver #数据增强
import random
# 数据增强
image_num=int(2000)#输入需要数据增强的数量
path='../tmp/class_data/'
for cl in list(os.listdir(path)):
    #获取类别图像路径
    adrss = os.path.join(path,cl)
    #读取路径下的所有图片
    images = os.listdir(adrss)
    #查看每种类别文件夹的图像数量
    num = int(len(images))
    if num<=int(image_num):
        num_pre=(int(image_num)-num)/4
        i=0
        for i in range(int(num_pre)):
            image_name= random.sample(os.listdir(adrss), 4) #随机取4张图像进行增强
            
            image_name1=image_name[0]
            image1=cv2.imdecode(np.fromfile(adrss+'/'+str(image_name[0]), dtype=np.uint8),-1)
            hv_=horandver(image1)
            hv_name=image_name1[:-4]+'-'+'ho-'+str(i)+'.jpg'
            cv2.imencode('.jpg',hv_)[1].tofile(adrss+'/'+hv_name)
            
            image_name2=image_name[1]
            image2=cv2.imdecode(np.fromfile(adrss+'/'+str(image_name[1]), dtype=np.uint8),-1)
            hor_=horizon_flip(image2)
            hor_name=image_name2[:-4]+'-'+'ho-'+str(i)+'.jpg'
            cv2.imencode('.jpg',hor_)[1].tofile(adrss+'/'+hor_name)
            
            image_name3=image_name[2]
            image3=cv2.imdecode(np.fromfile(adrss+'/'+str(image_name[2]), dtype=np.uint8),-1)
            ver_=vertical_flip(image3)
            ver_name=image_name3[:-4]+'-'+'ve-'+str(i)+'.jpg'
            cv2.imencode('.jpg',ver_)[1].tofile(adrss+'/'+ver_name)
            
            image_name4=image_name[3]
            image4=cv2.imdecode(np.fromfile(adrss+'/'+str(image_name[3]), dtype=np.uint8),-1)
            ro_=rotate(image4)
            ro_name=image_name4[:-4]+'-'+'ro-'+str(i)+'.jpg'
            cv2.imencode('.jpg',ro_)[1].tofile(adrss+'/'+ro_name)
            
            i=i+1
    else:
        pass
    
#归一化、标签热编码等             
import pandas as pd
import os
import cv2
import numpy as np
from shutil import rmtree
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical # 用于one-hot编码
from keras.preprocessing.image import img_to_array

data_file = '../tmp/class_data'
#定义类别对应的标签,需要与类别文件夹名字一致
name_dic = {'深灰色泥岩':0, '黑色煤':1, '灰色细砂岩':2, '浅灰色细砂岩':3, 
            '深灰色粉砂质泥岩':4, '灰黑色泥岩' : 5, '灰色泥质粉砂岩':6}
#需要的图片尺寸输入大小
height=128
width=128
#类别个数
class_num = 7
#保存图像与标签数据的文件夹位置
data_save='../tmp/data'

image_list = []#定义一个保存图片的空列表
label_list = []#定义一个保存标签的空列表
#读取文件夹下的类别文件夹
for file in os.listdir(data_file):
    name = str(file)
    name_count = 0
    #循环读取类别文件夹下的图片
    for key in os.listdir(data_file +'/'+ file):
        name_count+=1
        #将图片所在地址依次添加到列表中
        image_list.append(data_file +'/' +file + '/' + key)
        #按照定义的类别与对应标签，给图片打标签
        label_list.append(name_dic[file])
#将图片地址和所对应类别标签合
temp = np.array([np.hstack(image_list), np.hstack(label_list)])
data = temp.transpose()  # 对数据进行转置
#取出train和test的图片的路径   X数据集
data_address = list(data[:, 0])

data_image = []
#依次循环读取图像地址并保存为数组形式
#对train图片进行循环并进行去除背景和更改图片大小
for m in range(len(data_address)):
    image = cv2.imdecode(np.fromfile(data_address[m], dtype=np.uint8),-1)
    re_img=cv2.resize(image,(height,width),interpolation=cv2.INTER_AREA)#默认缩放
    data_image.append(re_img)


X= np.array(data_image, dtype="float")
label = list(data[:, 1])#取出数据所属的类别
Y = [int(i) for i in label]#顺序循环取出所属类别并保存为list形式   
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2)

# 数据标准化：提高模型预测精准度，加快收敛
X_train = np.array(X_train, dtype="float") /255.0#数据归一化
X_test = np.array(X_test, dtype="float") /255.0#数据归一化

np.save(data_save+'/'+'Y_test(label)1.npy', Y_test)

Y_train = to_categorical(Y_train, num_classes=class_num)
Y_test = to_categorical(Y_test, num_classes=class_num)

#创建保存数据的文件夹
if not os.path.exists(data_save):
    os.makedirs(data_save)

np.save(data_save+'/'+'X_train1.npy', X_train)
np.save(data_save+'/'+"Y_train1.npy", Y_train)

np.save(data_save+'/'+"X_test1.npy", X_test)
np.save(data_save+'/'+"Y_test1.npy", Y_test)




 




