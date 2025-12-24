import os
import numpy as np
import PIL
from PIL import Image,ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader
import torch
from einops import rearrange
import cv2 

class InpaintingBase(Dataset):
    def __init__(self,
                 csv_file,
                 data_root,
                 partition,
                 size,
                 interpolation="bicubic",
                 ):
        self.csv_df = pd.read_csv(csv_file)#读取csv文件
        self.csv_df = self.csv_df[self.csv_df["partition"] == partition]  # 筛选指定分区的数据
        self._length = len(self.csv_df) #获取筛选后的数据长度
        self.data_root = data_root#数据根目录
        self.size = size#图像大小
        self.transform = None # 图像转换操作，初始设为 None
        self.transform_mask = None # 掩码转换操作，初始设为 None
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation] # 插值方法，根据传入参数选择
        self.image_paths = self.csv_df["image_path"] # 图像路径
        self.mask_image = self.csv_df["mask_path"] # 掩码路径
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],#相对图像文件路径
            "file_path_": [os.path.join(self.data_root, l) #绝对图像文件路径
                           for l in self.image_paths],
            "relative_file_path_mask_": [l for l in self.mask_image], #相对掩码文件路径
            "file_path_mask_": [os.path.join(self.data_root, l) #绝对掩码文件路径
                           for l in self.mask_image],
        }

    def __len__(self):
        return self._length

    def _transform_and_normalize(self, image_path, mask_path):
        #读取和处理图像以及掩码图像，并进行转换和归一化处理
        image = Image.open(image_path).convert("RGB") ## 打开图像文件并转换为 RGB 模式
        mask = Image.open(mask_path) #打开掩码图像文件
        pil_mask = mask.convert('L') #将掩码图像转换为灰度图像
        image = self.transform(image) #对图像进行预定义的转换操作
        pil_mask = self.transform_mask(pil_mask) #对掩码图像进行预定义的转换操作
        masked_image = (1-pil_mask)*image #生成掩码后的图像
        return image, masked_image, pil_mask #返回原图、掩码图和处理后的掩码图

    def _transform_and_normalize_inference(self, image_path, mask_path, resize_to):
        #在推理阶段对图像和掩码进行转换和归一化处理
        image = np.array(Image.open(image_path).convert("RGB"))#打开图像文件并转换为 RGB 模式，转为 numpy 数组
        if image.shape[0]!=resize_to or image.shape[1] != resize_to:#调整图像尺寸
            image = cv2.resize(src=image, dsize=(resize_to,resize_to), interpolation = cv2.INTER_AREA)
        image = image.astype(np.float32)/255.0#将图像转换为 float32 类型，并归一化到 [0, 1] 范围
        image = image[None].transpose(0,3,1,2) #调整图像维度顺序
        image = torch.from_numpy(image) #将 numpy 数组转换为 PyTorch 张量
        mask = np.array(Image.open(mask_path).convert("L")) #打开掩码图像文件，并转换为灰度图像，转为 numpy 数组
        if mask.shape[0]!=resize_to or mask.shape[1]!=resize_to:#调整掩码图像尺寸
            mask = cv2.resize(src=mask, dsize=(resize_to,resize_to), interpolation = cv2.INTER_AREA)
        mask = mask.astype(np.float32)/255.0#将掩码图像转换为 float32 类型，并归一化到 [0, 1] 范围
        mask = mask[None,None] #调整掩码图像维度顺序
        mask[mask < 0.5] = 0 #将掩码值小于 0.5 的部分设为 0
        mask[mask >= 0.5] = 1 #将掩码值大于等于 0.5 的部分设为 1
        mask = torch.from_numpy(mask) #将 numpy 数组转换为 PyTorch 张量
        masked_image = (1-mask)*image #生成掩码后的图像
        batch = {"image": image, "mask": mask, "masked_image": masked_image} #创建包含图像、掩码和掩码图像的字典
        for k in batch:
            batch[k] = batch[k]*2.0-1.0 #将数据范围调整到 [-1, 1]
            if k=="mask":
                batch[k] = torch.squeeze(batch[k], dim=1) #压缩掩码张量 we are in get item here, so one at a time
            else:
                batch[k] = torch.squeeze(batch[k], dim=0)#压缩图像和掩码图像张量
            batch[k] = rearrange(batch[k], 'c h w -> h w c') #调整张量维度顺序
        return batch#返回处理后的字典

    def __getitem__(self, i):
        #获取数据集中的一个样本，并对图像和掩码进行处理。这个方法通常在数据加载器中被调用，用于在训练或推理过程中按索引获取样本
        example2 = dict((k, self.labels[k][i]) for k in self.labels) #创建一个包含所有标签键值对的字典
        add_dict = self._transform_and_normalize_inference(example2["file_path_"],
                                                           example2["file_path_mask_"],
                                                           resize_to=self.size)#调用 _transform_and_normalize_inference 方法，对图像和掩码进行处理
        example2.update(add_dict)#更新字典 example2，添加处理后的图像数据
        return example2#返回包含原始和处理后的图像数据的字典

class InpaintingTrain(InpaintingBase):#处理训练数据
    def __init__(self, csv_file, data_root, **kwargs):
        super().__init__(csv_file=csv_file, partition="train", data_root=data_root, **kwargs)#调用父类的构造函数，指定分区为 "train"
        self.transform = transforms.Compose([
                transforms.Resize((self.size,self.size)),#调整图像尺寸
                transforms.ToTensor(),#将图像转换为张量
        ])#定义图像转换操作
        self.transform_mask = transforms.Compose([
                transforms.Resize((self.size,self.size)),#调整掩码图像尺寸
                transforms.ToTensor(),#将掩码图像转换为张量
        ])#定义掩码图像转换操作

class InpaintingValidation(InpaintingBase):#用于处理验证数据
    def __init__(self, csv_file,data_root, **kwargs):
        super().__init__(csv_file=csv_file, partition="train", data_root=data_root, **kwargs)#调用父类的构造函数，指定分区为 "train"
        self.transform = transforms.Compose([
                        transforms.Resize((self.size, self.size)),
                        transforms.ToTensor(),
        ])
        self.transform_mask = transforms.Compose([
                transforms.Resize((self.size,self.size)),
                transforms.ToTensor(),
        ])

if __name__=="__main__":#设置图像和掩码的转换操作，加载训练数据，并对数据进行处理和保存
    size = 256
    transform = transforms.Compose([
        transforms.Resize((size,size)),#调整图像尺寸
        transforms.ToTensor(),#将图像转换为张量
    ])
    de_transform =  transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/255, 1/255 ,1/255 ]),#反归一化操作
                    ])
    
    de_transform_mask =  transforms.Compose([ transforms.Normalize(mean = [ 0. ],
                                                     std = [ 1/255]),#掩码图像反归一化操作
                    ])
    csv_file = "../data/INPAINTING/example_df.csv"
    data_root = "../data/INPAINTING/custom_inpainting"
    ip_train = InpaintingTrain(csv_file=csv_file, data_root=data_root, size= 256)#初始化训练数据
    ip_train_loader = DataLoader(ip_train, batch_size=1, num_workers=4,
                          pin_memory=True, shuffle=True)#加载训练数据
    for idx, batch in enumerate(ip_train_loader):
        im_keys = ['image', 'masked_image', 'mask']
        for k in im_keys:
            image_de = batch[k]
            image_de = (image_de + 1)/2#归一化处理
            image_de = rearrange(image_de, 'b h w c ->b c h w')#调整图像维度顺序
            if k=="mask":
                image_de = de_transform_mask(image_de)#反归一化掩码图像
            else:
                image_de = de_transform(image_de)#反归一化图像
            # 'b c h w ->b h w c'
            rgb_img = (image_de).type(torch.uint8).squeeze(0)#转换为无符号 8 位整数并去除批量维度
            img = transforms.ToPILImage()(rgb_img)#转换为 PIL 图像
            img.save("../data/test_loader_inpaint/%s_test.jpg" % k)#保存图像