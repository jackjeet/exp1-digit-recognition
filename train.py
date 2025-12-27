import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_loader import train_loader, test_loader
from model import MnistModel

# 全局变量：记录训练过程的损失和准确率
train_losses = []
train_accs = []
test_accs = []

# 1. 实例化模型、损失函数、优化器
model = MnistModel()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器


# 2. 模型训练函数
def train_model(epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 累计损失和准确率（强制转换为长整型避免bool错误）
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).long().sum().item()  # 关键修正

            # 打印中间结果（拆分长行）
            if batch_idx % 100 == 99:
                print(
                    f'Epoch [{epoch+1}/{epochs}], '
                    f'Batch [{batch_idx+1}/{len(train_loader)}], '
                    f'Loss: {running_loss/100:.4f}'
                )
                running_loss = 0.0

        # 记录训练指标
        train_losses.append(running_loss / len(train_loader))
        train_accs.append(100 * correct / total)
        test_accs.append(test_model_for_plot())

    print('训练完成！')


# 3. 辅助测试函数（仅返回准确率）
def test_model_for_plot():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).long().sum().item()  # 关键修正
    return 100 * correct / total


# 4. 详细评估函数
def test_model():
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    misclassified_examples = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).long().sum().item()  # 关键修正

            # 计算每类准确率（避免squeeze对单样本的影响）
            for label, pred in zip(labels, predicted):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

            # 收集错误案例（最多5个）
            if len(misclassified_examples) < 5:
                wrong_mask = (predicted != labels)
                if wrong_mask.sum().item() > 0:  # 用.item()转为Python数字
                    wrong_indices = torch.where(wrong_mask)[0]
                    for idx in wrong_indices[:5 - len(misclassified_examples)]:
                        misclassified_examples.append(
                            (images[idx], labels[idx], predicted[idx])
                        )

    # 打印评估结果
    print(f'\n测试集总体准确率：{100 * correct / total:.2f}%')
    print("每类数字识别准确率：")
    for i in range(10):
        print(f'数字 {i}: {100 * class_correct[i] / class_total[i]:.2f}%')

    return misclassified_examples


# 5. 模型保存
def save_model(path='mnist_model.pth'):
    torch.save(model.state_dict(), path)
    print(f'\n模型已保存至：{path}')


# 6. 模型加载与推理（暂时移除可视化，避免matplotlib问题）
def load_model_and_infer(path='mnist_model.pth'):
    loaded_model = MnistModel()
    loaded_model.load_state_dict(torch.load(path))
    loaded_model.eval()

    # 随机取测试集样本
    test_dataset = test_loader.dataset
    idx = np.random.randint(0, len(test_dataset))
    img, true_label = test_dataset[idx]
    img = img.unsqueeze(0)

    with torch.no_grad():
        output = loaded_model(img)
        _, pred_label = torch.max(output, 1)

    # 仅打印结果，不绘图
    print(f'\n随机样本推理：真实标签={true_label}，预测标签={pred_label.item()}')


# 主函数（移除matplotlib相关代码，确保能运行）
if __name__ == '__main__':
    # 先确保matplotlib已安装（若仍报错，可手动在终端运行安装命令）
    try:
        import matplotlib
    except ImportError:
        import os
        print("正在安装matplotlib...")
        os.system('pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple')

    # 执行核心流程（暂时不绘图，避免可视化错误）
    train_model(epochs=5)
    misclassified = test_model()
    save_model()
    load_model_and_infer()

# 文件末尾空行（符合PEP8）