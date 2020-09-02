import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片（单张image和label）
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        print("label_im_show", label)
        # cv2.imshow("label_im", label)
        # cv2.waitKey(0)
        # print("im_type", label.shape)
        # im_type (512, 512, 3)
        
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("label_dan", label)
        # cv2.waitKey(0)
        # label三通道和单通道展示出来的图差不多

        # print("label_gray", label.shape)
        # label_gray (512, 512)
        # label_reshape (1, 512, 512)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # print("re_label = {}".format(label.dtype))
        # re_label = uint8

        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        # print("label_shapef", label.shape)
        # label_shapef (1, 512, 512)
        #print("label_type = {}".format(label.dtype))
        # label_type = float64
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

    
if __name__ == "__main__":
    # 初始化调用__init__
    isbi_dataset = ISBI_Loader("data/train/")
    # len()实际调用了__len__
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2, 
                                               shuffle=True) 
    print("train_loader", train_loader)
    # 原来for in运算符实际上调用了__getitem__
    for image, label in train_loader:
        # torch.Size([2, 1, 512, 512])
        print(image.shape) 
   
