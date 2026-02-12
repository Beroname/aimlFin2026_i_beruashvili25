import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Very simple CNN example
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # adjust based on input size
        self.fc2 = nn.Linear(128, 2)             # example: 2 classes (benign/malicious)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()
print(model)

# Fake data just to generate a plot (replace with real images later)
fake_image = torch.rand(1, 1, 64, 64)  # batch=1, channels=1, 64x64
img_np = fake_image.squeeze().numpy()

plt.imshow(img_np, cmap='gray')
plt.title("Sample Grayscale Malware Image (simulated)")
plt.axis('off')
plt.savefig("malware_sample.png")
plt.show()

print("Image saved as malware_sample.png – upload this to task_1 folder")