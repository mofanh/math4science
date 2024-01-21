import numpy as np
import matplotlib.pyplot as plt
import wandb

# 初始化Wandb
wandb.init(
    project="SGD",
    config={
        "learning_rate": 0.01,
        "n_iterations": 10000,
        "m": 100
    }
)

# 随机X纬度x1，rand是随机均匀分布
x = 2 * np.random.rand(100, 1)
# 人为设置真实的Y一列
y = 4 + 3 * x + np.random.randn(100, 1)
# 整合 x0 和 x1 成矩阵
x_b = np.c_[np.ones((100, 1)), x]

# 设置初始参数和学习率
theta = np.random.randn(2, 1)  # 初始化theta
learning_rate = 0.01
n_iterations = 10000  # 迭代次数
m = 100  # 100 行

# 使用SGD进行优化
for iteration in range(n_iterations):
    # 随机索引，抽取出来，进行训练
    index = np.random.randint(m)
    xi = x_b[index:index + 1]
    yi = y[index:index + 1]
    
    # 计算梯度
    gradients = xi.T.dot(xi.dot(theta) - yi)
    
    # 更新theta值
    theta = theta - learning_rate * gradients
    
    # 记录到Wandb
    wandb.log({"iteration": iteration, "theta": theta, "loss": theta[1]})
    
    # 打印当前迭代的参数和损失
    # print(f'Iteration {iteration}, theta: {theta}, loss: {f(theta[1])}')

# 结束Wandb
wandb.finish()