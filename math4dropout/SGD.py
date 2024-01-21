import numpy as np
import matplotlib.pyplot as plt
import wandb

# 初始化Wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="SGD",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# 定义二次函数及其梯度
def f(x):
    return (x - 3)**2

def df(x):
    return 2 * (x - 3)

# 设置初始参数和学习率
x = 0.0
learning_rate = 0.1
num_iterations = 1000

# 使用SGD进行优化
for i in range(num_iterations):
    # 计算梯度
    gradient = df(x)
    
    # 更新参数
    x -= learning_rate * gradient
    
    # 记录到Wandb
    wandb.log({"iteration": i, "x": x, "loss": f(x)})

# 结束Wandb
wandb.finish()