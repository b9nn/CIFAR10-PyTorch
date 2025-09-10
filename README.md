# 🖼️ Image Classification with Transfer Learning (CIFAR-10)

## 📌 About the Project
This project applies **transfer learning** to classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), which contains **60,000 32×32 color images** across **10 object categories** (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

Instead of training a deep model from scratch, I fine-tuned a **ResNet18** pretrained on ImageNet. Transfer learning allows the model to leverage powerful feature extraction from large-scale datasets, making training faster and improving accuracy.

The project demonstrates how to:
- Load and preprocess CIFAR-10 images  
- Apply **transfer learning** using PyTorch  
- Fine-tune a pretrained model (ResNet18)  
- Evaluate performance with **accuracy and loss curves**  

---

## ⚙️ Tech Stack
- Python 3.10+  
- PyTorch  
- Torchvision  

---

## 🚀 Features
- ✅ Transfer learning with ResNet18  
- ✅ CIFAR-10 dataset support  
- ✅ GPU acceleration (CUDA)  
- ✅ Training/validation loss & accuracy tracking  

---

## 📊 Results
- Achieved **high accuracy** on CIFAR-10 using fine-tuned ResNet18  
- Reduced training time compared to training from scratch  
- Robust generalization on unseen test data  



---

## 📂 Project Structure
