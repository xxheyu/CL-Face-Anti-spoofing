import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

train_root = '/export/data/zqs/Data/HiFiMask-Challenge/phase1'
test_root = '/opt/cephfs_workspace/gpudisk/Qiusheng/Data/HiFiMask-Challenge/phase2'

# -----------------ready the dataset--------------------------
def default_loader(path, root):
    if path.split('/')[0] == 'test':
        path = os.path.join(root, path)
    elif path.split('/')[0] == 'train':
        path = os.path.join(root, path)
    else:
        path = os.path.join(root, 'val', path)
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    # 构造函数带有默认参数
    def __init__(self, root, txt, transform=None, target_transform=None, loader=default_loader, train=True):
        self.root = root
        self.path = os.path.join(root, txt)
        fh = open(self.path, 'r')
        imgs = []
        self.train = train
        if self.train:
            for line in fh:
                # 移除字符串首尾的换行符
                # 删除末尾空
                # 以空格为分隔符 将字符串分成
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                imgs.append((words[0], int(words[1])))  # imgs中包含有图像路径和标签
        else:
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                imgs.append((words[0]))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        if self.train:
            fn, label = self.imgs[index]
            # 调用定义的loader方法
            img = self.loader(fn, self.root)
            if self.transform is not None:
                img = self.transform(img)
            return img, label, fn
        else:
            fn = self.imgs[index]
            # 调用定义的loader方法
            img = self.loader(fn, self.root)
            if self.transform is not None:
                img = self.transform(img)
            return img, fn

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    #train_data = MyDataset(txt=root + 'train_label.txt', transform=transforms.ToTensor())
    test_data = MyDataset(txt=os.path.join(test_root, 'test.txt'), transform=transforms.ToTensor(), train=False)

    # # train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
    # train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    # images = []
    # labels = []
    # for image, label in train_loader:
    #     images.append(image)
    #     labels.append(label)
    # print('***********')
    test_loader = DataLoader(dataset=test_data, batch_size=64)
    images = []
    labels = []
    for image, label in test_loader:
        images.append(image)
        labels.append(label)
