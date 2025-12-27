import torch
from PIL import Image  # 用于读取图片
import numpy as np
from torchvision import transforms  # 用于图片预处理
from model import MnistModel  # 导入模型结构

# 1. 图片预处理（和训练时保持一致，否则预测不准）
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整为28×28像素（MNIST图片大小）
    transforms.Grayscale(),       # 转为黑白图（MNIST是单通道）
    transforms.ToTensor(),        # 转为PyTorch能识别的张量
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化（固定值，和训练时一致）
])

# 2. 加载预训练模型
def load_model(path='mnist_model.pth'):
    model = MnistModel()  # 创建模型结构（必须和训练时一样）
    model.load_state_dict(torch.load(path))  # 加载保存的参数
    model.eval()  # 切换到评估模式（重要！）
    return model

# 3. 预测单张图片
def predict(model, image_path):
    # 读取图片
    img = Image.open(image_path)
    # 预处理图片
    img = transform(img)
    # 增加一个批次维度（模型要求输入格式：[1, 1, 28, 28]）
    img = img.unsqueeze(0)

    # 开始预测（不计算梯度，节省资源）
    with torch.no_grad():
        output = model(img)  # 模型输出10个数字的得分
        _, pred = torch.max(output, 1)  # 取得分最高的数字

    return pred.item()  # 返回预测的数字

# 4. 主函数：执行加载和预测
if __name__ == '__main__':
    # 加载模型（如果.pth文件在其他位置，改路径即可）
    my_model = load_model()
    print("模型加载成功！")

    # 预测你准备的图片（替换为你的图片文件名，比如"test.png"）
    image_path = "5.png"  # 图片必须和infer.py在同一个文件夹
    try:
        result = predict(my_model, image_path)
        print(f"预测结果：这张图片是数字 {result}")
    except:
        print("出错了！请检查图片是否存在，或图片格式是否正确（png/jpg均可）")