# 🌍 Satellite Image Classification

### Comparative Deep Learning Study Using CNNs and Vision Transformers (Keras & PyTorch)

This repository presents a comprehensive deep learning framework for
**satellite image classification**, implementing and evaluating models
in both **Keras (TensorFlow)** and **PyTorch**.

The project explores:

-   CNN-based architectures\
-   Vision Transformers (ViT)\
-   Cross-framework implementation (Keras vs PyTorch)\
-   Training pipelines\
-   Evaluation metrics & comparative analysis

------------------------------------------------------------------------

## 📂 Repository Structure

    Satellite-Image-Classification/
    │
    ├── Satellite-Image-Classification- Keras.ipynb
    ├── Satellite-Image-Classification- Train- Keras.ipynb
    ├── Satellite-Image-Classification- Vision Transformers Using Keras.ipynb
    │
    ├── Satellite-Image-Classification- Pytorch.ipynb
    ├── Satellite-Image-Classification- Train- Pytorch.ipynb
    ├── Satellite-Image-Classification- Vision Transformers Using PyTorch.ipynb
    │
    ├── CNN-Transformer Integration evaluation- ViT Model Evaluation.ipynb
    ├── Satellite-Image-Classification- Comparative Analysis of Keras and PyTorch Models.ipynb
    │
    └── README.md

------------------------------------------------------------------------

## 🚀 Project Overview

Satellite imagery classification plays a critical role in:

-   Land-use detection\
-   Urban planning\
-   Environmental monitoring\
-   Agricultural analysis

This project compares traditional CNN-based architectures with modern
**Vision Transformers (ViT)** to evaluate:

-   Accuracy\
-   Generalization\
-   Computational performance\
-   Framework efficiency

------------------------------------------------------------------------

## 🧠 Implemented Models

### 🔹 CNN-Based Models

-   Built in both **Keras** and **PyTorch**
-   Standard convolutional architecture
-   Cross-entropy loss
-   Softmax classification

### 🔹 Vision Transformer (ViT)

-   Patch embedding layer
-   Positional encoding
-   Multi-head self-attention
-   Transformer encoder blocks
-   Classification head

Implemented in:

-   `Satellite-Image-Classification- Vision Transformers Using Keras.ipynb`
-   `Satellite-Image-Classification- Vision Transformers Using PyTorch.ipynb`

------------------------------------------------------------------------

## 📊 Training Pipelines

Training notebooks include:

-   Data preprocessing\
-   Data augmentation\
-   Model definition\
-   Loss & optimizer setup\
-   Training loop\
-   Validation monitoring\
-   Model saving

Files:

-   `Satellite-Image-Classification- Train- Keras.ipynb`
-   `Satellite-Image-Classification- Train- Pytorch.ipynb`

------------------------------------------------------------------------

## 📈 Evaluation

Comprehensive evaluation metrics include:

-   Accuracy\
-   Precision\
-   Recall\
-   F1-score\
-   ROC-AUC\
-   Log Loss\
-   Confusion Matrix\
-   Classification Report

Evaluation notebook:

-   `CNN-Transformer Integration evaluation- ViT Model Evaluation.ipynb`

------------------------------------------------------------------------

## 🔬 Comparative Analysis

The notebook:

`Satellite-Image-Classification- Comparative Analysis of Keras and PyTorch Models.ipynb`

provides:

-   Performance comparison between frameworks\
-   CNN vs Transformer analysis\
-   Training time comparison\
-   Metric breakdown\
-   Observations and insights

------------------------------------------------------------------------

## ⚙️ Technologies Used

-   Python 3.x\
-   TensorFlow / Keras\
-   PyTorch\
-   NumPy\
-   Scikit-learn\
-   Matplotlib / Seaborn

------------------------------------------------------------------------

## 🏗 Model Architecture Summary

### Vision Transformer Pipeline

    Input Image
       ↓
    Patch Embedding
       ↓
    Positional Encoding
       ↓
    Transformer Encoder Blocks
       ↓
    Classification Head
       ↓
    Softmax Output

------------------------------------------------------------------------

## 📦 Installation

``` bash
git clone https://github.com/your-username/Satellite-Image-Classification.git
cd Satellite-Image-Classification
pip install -r requirements.txt
```

Or manually install:

``` bash
pip install tensorflow torch torchvision numpy scikit-learn matplotlib
```

------------------------------------------------------------------------

## ▶️ How to Run

### Keras

Open:

`Satellite-Image-Classification- Train- Keras.ipynb`

### PyTorch

Open:

`Satellite-Image-Classification- Train- Pytorch.ipynb`

------------------------------------------------------------------------

## 🎯 Key Contributions

-   Dual-framework implementation (Keras & PyTorch)\
-   Transformer-based satellite classification\
-   Full evaluation pipeline\
-   Reproducible training workflows\
-   Comparative deep learning study

------------------------------------------------------------------------

## 📌 Future Improvements

-   Hybrid CNN + ViT architecture\
-   Transfer learning with pretrained ViT\
-   Larger satellite datasets\
-   Model optimization for deployment

------------------------------------------------------------------------

## 📄 License

This project is released under the MIT License.

------------------------------------------------------------------------

## 👨‍💻 Author

Aref Aasi