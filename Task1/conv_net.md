# Task 1 – Convolutional Neural Networks (CNN)

## Description of Convolutional Neural Networks

A **Convolutional Neural Network (CNN)** is a class of deep neural networks most commonly applied to analyzing visual imagery. CNNs are designed to automatically and adaptively learn spatial hierarchies of features through backpropagation by using multiple building blocks: convolutional layers, pooling layers, and fully connected layers.

The core innovation of CNNs is the **convolution operation**. Instead of connecting every neuron to every neuron in the previous layer (as in fully connected networks), a convolutional layer applies a small learnable filter (kernel) that slides over the input volume. At each position, it computes a dot product between the filter weights and the input patch, producing a feature map. This operation allows the network to detect local patterns (edges, textures, shapes) regardless of their exact position in the image — a property called **translation equivariance**.

Multiple filters are used in parallel, each learning to detect different features. After convolution, a non-linear activation function (most commonly ReLU) is applied. **Pooling layers** (usually max-pooling) then reduce the spatial dimensions of the feature maps, making the representation more compact and providing some translation invariance. Finally, after several convolutional + pooling blocks, the feature maps are flattened and passed through one or more **fully connected (dense) layers** for classification or regression.

Key advantages of CNNs include:

- Drastically fewer parameters due to weight sharing and sparse connectivity
- Hierarchical feature learning (low-level → high-level features)
- Strong performance on grid-like data (images, spectrograms, time-series)

CNNs form the backbone of modern computer vision: image classification (AlexNet, ResNet, EfficientNet), object detection (YOLO, Faster R-CNN), segmentation, face recognition, and many cybersecurity applications.

(Word count: ~340)

## Visualizations

![CNN architecture overview](https://miro.medium.com/v2/resize:fit:1400/1*BMng6fQ4ht8k4c7jG4d2dA.png)  
*Typical CNN flow: conv → pool → conv → pool → fully connected*

![Convolution operation](https://media.geeksforgeeks.org/wp-content/uploads/20210903162947/conv.gif)  
*How a 3×3 kernel slides over the input and produces a feature map*

![Max pooling](https://media.geeksforgeeks.org/wp-content/uploads/20210903163531/maxpooling.png)  
*Max pooling reduces spatial size while preserving important information*

## Practical Example in Cybersecurity: Malware Image Classification

One well-known application of CNNs in cybersecurity is **malware family classification** using the **MalImg** dataset. Malware executables are converted into grayscale images (byte value → pixel intensity), revealing visual patterns in code structure, entropy, and repeating sequences. CNNs can then classify the malware family with high accuracy.

**Example code** (simple CNN using Keras / TensorFlow):

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Simulated data: in real case load MalImg images (resize to 64x64 grayscale)
# Here we create dummy data for demonstration
np.random.seed(42)
X = np.random.rand(200, 64, 64, 1)          # 200 samples
y = np.random.randint(0, 9, 200)            # 9 classes (example)

# Build a simple CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(9, activation='softmax')   # 9 malware families example
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train (in practice use real data + data augmentation)
# model.fit(X, y, epochs=10, validation_split=0.2)

# Example visualization
plt.figure(figsize=(4,4))
plt.imshow(X[0].reshape(64,64), cmap='gray')
plt.title("Example Malware Image (simulated)")
plt.axis('off')
plt.savefig("malware_image_example.png")
plt.show()