from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, path , image_paths, transform=None):
        self.path = path
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.path+self.image_paths[idx])
        if self.transform:
            img = self.transform(img)
        return img



class ImageVectorizer:
    def __init__(self, path=None):
        # 加载预训练模型
        self.model = models.resnet34(weights='ResNet34_Weights.DEFAULT')
        self.model = self.model.to("cuda:0")
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))  # 移除最后一层
        self.model.eval()
        self.batch_size = 1024
        self.path = path

        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.nopreprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

    def image_to_vector(self, image_path):
        img = Image.open(image_path)
        img_t = self.preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0).to("cuda:0")

        with torch.no_grad():
            features = self.model(batch_t)

        return features.squeeze().to('cpu').numpy()

    def images_to_vector(self, num):
        """
        批量处理前num个图片并返回向量结果

        参数:
            num (int): 要处理的图片数量

        返回:
            numpy.ndarray: 形状为(num, feature_dim)的数组，包含所有图片的特征向量
        """
        # 准备图片路径列表
        dir = self.path
        # image_paths = [f"../archive/data/{i}.png" for i in range( last + 1 , last + num + 1)]
        # 获取目录下所有文件和子目录名
        image_paths  = os.listdir(dir)
        image_paths = image_paths[:num]
        print(image_paths[:10])
        # 创建数据集和数据加载器
        dataset = ImageDataset(image_paths=image_paths, path=self.path,transform=self.preprocess)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_features = []

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to("cuda:0")
                features = self.model(batch)
                features = features.squeeze()  # 移除不必要的维度
                all_features.append(features.to("cpu").numpy())

        # 合并所有batch的结果
        features_array = np.concatenate(all_features, axis=0)

        return features_array

    # def image_to_numpy(self, image_path):
    #     img = Image.open(image_path)
    #     img_t = self.nopreprocess(img)
    #     batch_t = torch.unsqueeze(img_t, 0)

    #     return batch_t.squeeze().numpy()

    # def images_to_numpy(self, num, last = 0):
    #     """
    #     批量处理前num个图片并返回向量结果

    #     参数:
    #         num (int): 要处理的图片数量

    #     返回:
    #         numpy.ndarray: 形状为(num, feature_dim)的数组，包含所有图片的特征向量
    #     """
    #     # 准备图片路径列表
    #     image_paths = [f"../archive/data/{i}.png" for i in range(last + 1, last + num + 1)]

    #     # 过滤掉不存在的图片路径
    #     valid_paths = []
    #     for path in image_paths:
    #         try:
    #             with Image.open(path) as img:
    #                 valid_paths.append(path)
    #         except:
    #             print(f"Warning: Image {path} not found or corrupted, skipping")
    #             continue

    #     # 创建数据集和数据加载器
    #     dataset = ImageDataset(valid_paths, transform=self.nopreprocess)
    #     dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    #     all_features = []

    #     with torch.no_grad():
    #         for batch in dataloader:
    #             batch = batch.squeeze()  # 移除不必要的维度
    #             all_features.append(batch.numpy())

    #     # 合并所有batch的结果
    #     features_array = np.concatenate(all_features, axis=0)

    #     return features_array