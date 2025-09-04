Fashion-MNIST Image Classification 👕👖👟👜

This project implements image classification on the Fashion-MNIST dataset
 using Neural Networks (NN) and Convolutional Neural Networks (CNN).

📂 Dataset

Fashion-MNIST consists of 70,000 grayscale images (28x28 pixels).

10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

Train set: 60,000 images
Test set: 10,000 images

🧠 Approaches
1. Fully Connected Neural Network (NN)

Input: Flattened 28×28 images → 784 features

Architecture:

Linear(784 → 256) + ReLU
Linear(256 → 32) + ReLU
Linear(32 → 10) + Softmax

Achieved accuracy: ~ 86%


2. Convolutional Neural Network (CNN)

Input: 1×28×28 image
Architecture:

Conv2D(1 → 32, kernel=3) + ReLU + MaxPool
Conv2D(32 → 64, kernel=3) + ReLU + MaxPool
Flatten
Linear(64*7*7 → 128) + ReLU
Linear(128 → 10) + Softmax

Achieved accuracy: ~ 90.34%


3. Mini-AlexNet for Fashion-MNIST

Input: 1×28×28 grayscale image

Architecture:

Conv Block 1: Conv2D(1 → 64, kernel=3, stride=1, padding=1) → BatchNorm → ReLU → MaxPool(2×2)
Conv Block 2: Conv2D(64 → 128, kernel=3, stride=1, padding=1) → BatchNorm → ReLU → MaxPool(2×2)
Conv Block 3: Conv2D(128 → 256, kernel=3, stride=1, padding=1) → BatchNorm → ReLU → MaxPool(2×2)

Flatten: 256 × 3 × 3 → vector
Fully Connected 1: 256×3×3 → 512 → ReLU → Dropout(0.5)
Fully Connected 2: 512 → 10 → logits (for CrossEntropyLoss)

Notes:

Pooling: Each MaxPool halves spatial dimensions: 28 → 14 → 7 → 3
BatchNorm: Stabilizes training and speeds up convergence
Dropout: 50% probability to reduce overfitting

Achieved accuracy: ~ 91.55%
