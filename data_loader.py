# 1. 导入需要的库
import torch
from torchvision import datasets, transforms  # 加载MNIST和预处理工具
from torch.utils.data import DataLoader  # 批处理工具

# 2. 数据预处理：转为张量（PyTorch能识别的格式）+ 归一化
# （这一步是实验步骤2要求的“数据预处理”）
transform = transforms.Compose([
    transforms.ToTensor(),  # 图片转张量（形状：[1,28,28]）
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化（MNIST官方推荐值）
])

# 3. 加载数据集并划分训练集/测试集
# （这一步是实验步骤2要求的“选择数据集、划分训练集和测试集”）
# root：指向我们创建的data文件夹
# train=True → 训练集；train=False → 测试集
# download=False → 已手动放好数据，不重复下载
train_dataset = datasets.MNIST(
    root='./data',  # 相对路径，对应上面创建的data文件夹
    train=True,
    transform=transform,
    download=False
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=False
)

# 4. 用DataLoader批处理和打乱数据
# （这一步是实验步骤2要求的“调用DataLoader处理数据”）
batch_size = 64  # 每次处理64张图片
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True  # 训练集打乱，增强泛化能力
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False  # 测试集不用打乱
)

# 5. 验证是否准备成功（打印关键信t10k-images-idx3-ubyte.gz息）
if __name__ == '__main__':
    print(f"训练集样本数：{len(train_dataset)}（预期60000）")
    print(f"测试集样本数：{len(test_dataset)}（预期10000）")
    # 取一个批次的数据，查看形状
    images, labels = next(iter(train_loader))
    print(f"一个批次的图片形状：{images.shape}（预期[64,1,28,28]）")
    print(f"一个批次的标签形状：{labels.shape}（预期[64]）")