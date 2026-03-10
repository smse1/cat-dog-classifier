# 🐱🐶 Cat Dog Classifier

一个基于 **PyTorch** 的猫狗二分类项目，实现了从 **数据读取、模型训练、验证集评估、测试集预测** 到 **`submission.csv` 生成** 的完整流程。项目采用 **ResNet18** 作为基础模型，并在本地 **RTX 5070 Laptop GPU** 上完成训练。

---

## 📌 项目简介

本项目用于实现 **Dogs vs Cats** 图像分类任务，目标是判断输入图片属于 **猫（cat）** 还是 **狗（dog）**。

当前版本已经实现：

- 数据集目录检查
- 训练集 / 验证集加载
- 图像预处理与数据增强
- 基于 ResNet18 的二分类模型
- GPU 训练与最优模型保存
- 测试集预测
- 生成符合提交要求的 `submission.csv`

---

## 🗂️ 项目结构

```text
cat-dog-classifier/
├── checkpoints/
│   └── best_model.pth
├── data/
│   └── README.md
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── utils.py
├── .gitignore
├── README.md
├── predict.py
├── requirements.txt
├── submission.csv
└── train.py
