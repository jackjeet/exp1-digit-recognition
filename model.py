import torch
import torch.nn as nn  # 导入PyTorch的神经网络模块


# 定义模型类，必须继承nn.Module
class MnistModel(nn.Module):
    def __init__(self):
        # 调用父类构造函数，固定写法
        super(MnistModel, self).__init__()

        # 定义网络层（输入层→隐藏层1→隐藏层2→输出层）
        # 1. 输入层：784（28×28）→ 256（隐藏层1神经元数）
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=256)
        # 2. 隐藏层1→隐藏层2：256→128
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        # 3. 隐藏层2→输出层：128→10（10个数字类别）
        self.fc3 = nn.Linear(in_features=128, out_features=10)

        # 定义激活函数（ReLU）和 dropout层（可选，防止过拟合）
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)  # 随机丢弃20%的神经元

    # 定义前向传播方法（核心，决定数据如何流过网络）
    def forward(self, x):
        # 输入x的形状：[batch_size, 1, 28, 28]（批次大小×通道×高×宽）
        # 第一步：展平图像（将28×28的二维图像转为1维向量）
        x = x.view(-1, 28 * 28)  # 展平后形状：[batch_size, 784]

        # 第二步：输入层→隐藏层1（加激活函数和dropout）
        x = self.fc1(x)  # 线性变换：784→256
        x = self.relu(x)  # 激活函数：引入非线性
        x = self.dropout(x)  # 防止过拟合

        # 第三步：隐藏层1→隐藏层2
        x = self.fc2(x)  # 线性变换：256→128
        x = self.relu(x)  # 激活函数

        # 第四步：隐藏层2→输出层（不激活，后续用交叉熵损失）
        x = self.fc3(x)  # 线性变换：128→10

        return x  # 输出形状：[batch_size, 10]（每个样本的10个类别得分）


# 测试模型是否能正常运行（可选，用于验证）
if __name__ == '__main__':
    # 创建模型实例
    model = MnistModel()
    # 生成一个随机的“批次数据”（模拟16张28×28的图片）
    test_input = torch.randn(16, 1, 28, 28)  # 形状：[16,1,28,28]
    # 前向传播
    output = model(test_input)
    # 打印输出形状（预期：[16,10]）
    print(f"模型输出形状：{output.shape}")  # 正确输出应为 torch.Size([16, 10])