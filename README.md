# 🧠 Deep Learning for Image Classification

This project implements and evaluates deep learning models on the **Imagenette** and **CIFAR10** datasets using PyTorch Lightning.  
It covers a basic CNN, ResNet-18, regularization techniques, and transfer learning.

---

## 📚 Project Overview

* **Framework**: PyTorch Lightning  
* **Datasets**: Imagenette, CIFAR10  
* **Platform**: Google Colab (CPU / GPU)

---

## 🛠 How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/deep-learning-imagenette-cifar10.git
   cd deep-learning-imagenette-cifar10
   ```

2. **Install dependencies**

   ```bash
   pip install pytorch-lightning torchvision matplotlib
   ```

3. **Run on Google Colab (recommended)**

   🔗 [Google Colab](https://colab.research.google.com/)

4. **Launch Jupyter notebook**

   ```bash
   jupyter notebook deep_learning_project.ipynb
   ```

---

## 💡 Features & Highlights

* Basic CNN with ~2.2M parameters  
* ResNet-18 architecture with 11.2M parameters  
* Data augmentation with random crop, flip, rotation, and grayscale  
* Transfer learning from Imagenette to CIFAR10  
* Early stopping to prevent overfitting  
* Training and validation loss tracking  
* Final test accuracy comparison

---

## 📊 Results Summary

### 📦 Basic CNN (Imagenette)

* **Architecture**
  * 3 convolutional layers + 2 fully connected layers
* **Training**
  * Optimizer: Adam (lr=1e-3), CrossEntropyLoss
  * Early stopping (patience=5)
* **Results**
  * Validation Accuracy: 59.6%
  * Test Accuracy: 60.99%
  * Test Loss: 1.97

---

### 📦 ResNet-18 (Imagenette)

* **Architecture**
  * Standard ResNet-18, trained from scratch
* **Results**
  * Validation Accuracy: 49.4%
  * Test Accuracy: 45.99%
  * Test Loss: 1.67

---

### 📦 ResNet-18 with Data Augmentation

* **Augmentations**
  * RandomResizedCrop, RandomHorizontalFlip, RandomRotation(10°), Grayscale, Normalize
* **Results**
  * Original ResNet-18 → Test Accuracy: 45.15%, Test Loss: 1.66  
  * With Augmentation → Test Accuracy: 57.45%, Test Loss: 1.95

✅ Accuracy improved by ~12% with regularization.

---

### 📦 Transfer Learning on CIFAR10

| Model Variant                      | Test Accuracy | Test Loss |
| ---------------------------------- | ------------- | --------- |
| Trained from Scratch               | 77.05%        | 0.7157    |
| Fine-Tuned (Imagenette Pretrained) | 80.04%        | 0.6070    |

✅ Transfer learning improved accuracy and generalization on CIFAR10.

---

## ✨ Dataset Links

* [Imagenette Dataset](https://github.com/fastai/imagenette)
* [CIFAR10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 📄 References

* [PyTorch Lightning](https://www.pytorchlightning.ai/)
* [Imagenette Dataset](https://github.com/fastai/imagenette)
* [CIFAR10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
* [Transfer Learning Demo](https://github.com/ajdillhoff/CSE6363/blob/main/deep_learning/transfer_learning.ipynb)
* ["Introduction to Decision Trees (Titanic dataset)"](https://www.kaggle.com/code/dmilla/introduction-to-decision-trees-titanic-dataset)

---

## 🎓 Key Learnings

* Building and tuning CNNs for image classification  
* Applying ResNet architectures from scratch  
* Using data augmentation to improve generalization  
* Leveraging transfer learning to boost performance on new datasets  
* Implementing early stopping to reduce overfitting

---

## 📜 License

© 2025 **Anamol Khadka**. All rights reserved.  
This work is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

You are free to **share** and **adapt** the material for any purpose, even commercially, as long as **appropriate credit** is given.

For inquiries, please contact: [khadkaanamol8@gmail.com](mailto:khadkaanamol8@gmail.com)