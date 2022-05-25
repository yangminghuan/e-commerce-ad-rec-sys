import torchvision
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import os
from PIL import Image

# 图片输入路径
input1_root = "./pic/input1/"
input2_root = "./pic/input2/"
label_root = "./pic/label/"
transforms_img = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


class MyDataset(Dataset):
    def __init__(self, input1, input2, label, transform=None):
        self.input1_root = input1
        self.input2_root = input2
        self.label_root = label

        # 读取路径下的所有图片文件名称列表
        self.input1_files = os.listdir(input1)
        self.input2_files = os.listdir(input2)
        self.label_files = os.listdir(label)

        self.transforms = transform

    def __len__(self):
        # 返回数据集大小
        return len(self.label_files)

    def __getitem__(self, index):
        # 根据索引index读取对应的图片
        input1_img_path = os.path.join(self.input1_root, self.input1_files[index])
        input1_img = Image.open(input1_img_path)

        input2_img_path = os.path.join(self.input2_root, self.input2_files[index])
        input2_img = Image.open(input2_img_path)

        label_img_path = os.path.join(self.label_root, self.label_files[index])
        label_img = Image.open(label_img_path)

        # 将图片转换为tensor类型
        if self.transforms:
            input1_img = self.transforms(input1_img)
            input2_img = self.transforms(input2_img)
            label_img = self.transforms(label_img)

        return (input1_img, input2_img), label_img


if __name__ == "__main__":
    dataset = MyDataset(input1_root, input2_root, label_root, transforms_img)
    loader = DataLoader(dataset)
    for input_data, output_data in loader:
        print(input_data[0].shape)
        print(input_data[1].shape)
        print(output_data.shape)
        break
