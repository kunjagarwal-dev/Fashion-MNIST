# 👕 Fashion MNIST Classification

A deep learning project comparing a baseline CNN, a regularized CNN, and MobileNetV2 transfer learning on Fashion MNIST, with TensorFlow Lite conversion for lightweight deployment, and an interactive Streamlit app.

## 📊 Results

| Model                                                 | Val Accuracy | Notes                                        |
| ----------------------------------------------------- | ------------ | -------------------------------------------- |
| Baseline CNN                                          | ~91%         | Mild overfitting appearing late in training  |
| Regularized CNN (BatchNorm + Dropout + EarlyStopping) | ~91.5–91.6%  | More stable val_loss curve, less overfitting |
| MobileNetV2 (transfer learning)                       | ~90.85%      | Comparable to custom CNNs, not better        |

**TFLite conversion:**

| Format                 | Size                        |
| ---------------------- | --------------------------- |
| Keras (.h5)            | 10.86 MB                    |
| TFLite (unquantized)   | 9.08 MB                     |
| **TFLite (quantized)** | **2.55 MB** (~4.3x smaller) |

Quantized model predictions were verified to exactly match the original Keras model's predictions on held-out test samples — no accuracy loss from quantization on the tested cases.

## 🗂️ Project Structure

```
fashion-mnist-classification/
├── notebooks/
│   ├── fashion_mnist.ipynb        # Baseline + regularized CNN
│   └── fashion_mobilenet.ipynb    # MobileNetV2 transfer learning + TFLite conversion
├── models/
│   ├── fashion_baseline_cnn.h5
│   ├── fashion_regularized_cnn.h5
│   ├── fashion_mobilenet.h5
│   └── fashion_mobilenet_quant.tflite
├── app/
│   └── streamlit_app.py
├── assets/
│   └── sample_grid.png
├── requirements.txt
├── .gitignore
└── README.md
```

## 🧠 Models

**Baseline CNN:** Two conv blocks (32, 64 filters), Flatten, Dense(128), softmax(10). Solid starting point but showed the beginning of an overfitting gap by the final epochs.

**Regularized CNN:** Added BatchNormalization after each conv layer and Dropout before the dense layers, plus `ReduceLROnPlateau` and `EarlyStopping` callbacks. Training stopped automatically at epoch 12, producing a noticeably more stable validation loss curve than the baseline, even though the raw accuracy gain was modest.

**MobileNetV2 (transfer learning):** A frozen, ImageNet-pretrained MobileNetV2 backbone with a custom classification head. Built using the Keras Functional API with `training=False` explicitly set on the frozen base — a fix learned from an earlier, unsuccessful ResNet50 transfer-learning attempt on CIFAR-10 where this exact detail caused a severe accuracy plateau. Applied correctly here, at an adequate 96×96 input resolution, this model trained cleanly and reached accuracy comparable to the custom CNNs — though interestingly, not better. This suggests that for simple, low-complexity grayscale datasets like Fashion MNIST, pretrained ImageNet features don't offer the same advantage they typically do on more complex, photo-realistic datasets.

**TFLite conversion:** The MobileNetV2 model was converted to TensorFlow Lite format, then further compressed via post-training quantization (32-bit float weights → 8-bit integers), reducing the deployable model size by roughly 4.3x with verified prediction consistency against the original model.

## 🚀 Running the App

```
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Upload a clothing image and get a live prediction from the quantized TFLite model, with a confidence breakdown across all 10 classes.

## 🛠️ Tech Stack

TensorFlow / Keras, TensorFlow Lite, Streamlit, NumPy, Matplotlib, Seaborn, scikit-learn

## 📈 Skills Demonstrated

- CNN architecture design, regularization (BatchNorm, Dropout, callbacks)
- Transfer learning with a pretrained backbone, using the Functional API to correctly handle frozen BatchNormalization behavior
- Diagnosing and applying a fix learned from a prior unresolved bug (CIFAR-10 ResNet50) to a new project successfully
- Model compression via TensorFlow Lite conversion and post-training quantization
- Verifying model correctness after format conversion
- Interactive model deployment with Streamlit
