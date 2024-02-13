import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 读取图像
img = Image.open('/home/lbj/桌面/front/web-front/src/assets/lier-logo.png')

# 定义变换
transform = transforms.Compose([
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# 应用变换
transformed_img = transform(img)

# 显示原始图像
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img)

# 显示变换后的图像
plt.subplot(1, 2, 2)
plt.title('Transformed Image')
plt.imshow(transformed_img.permute(1, 2, 0))  # 将张量转换为图像

plt.show()