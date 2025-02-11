
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.15, gamma=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# 示例模型（简单的多层感知器）
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 定义模型、损失函数和优化器（假设输入大小为10，输出类别为3）
# model = SimpleNN(10, 5, 3)
# criterion = FocalLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# # 示例输入和目标输出（随机数据）
# inputs = torch.randn(32, 10)
# targets = torch.randint(0, 3, (32,)).type(torch.FloatTensor)
#
# # 训练步骤
# optimizer.zero_grad()
# outputs = model(inputs)
# loss = criterion(outputs, targets)
# loss.backward()
# optimizer.step()
#
# print(f'Loss: {loss.item()}')
