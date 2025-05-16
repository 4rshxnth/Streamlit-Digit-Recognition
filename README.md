# 🧠 Streamlit Digit Recognition App

A simple and interactive web app for recognizing handwritten digits using a machine learning model trained on the `sklearn` Digits dataset. Built using Streamlit for seamless UI and deployed on Streamlit Community Cloud.

---

## 🚀 Demo

👉 [Live App](https://your-username-your-repo-name.streamlit.app)  
_(Replace with your deployed URL)_

---

## 🖼️ Features

- Upload a handwritten digit image (JPG/PNG)
- Real-time digit prediction using Support Vector Classifier (SVC)
- Visualize the processed 8x8 grayscale input
- Lightweight, fast, and cloud-deployable

---

## 🛠️ Tech Stack

- **Python 3**
- **Streamlit**
- **scikit-learn**
- **Matplotlib**
- **Pillow (PIL)**

---

## 🧪 Model Overview

- Dataset: `sklearn.datasets.load_digits` (0–9 digits)
- Model: Support Vector Classifier (`SVC`)
- Training/Testing: 80/20 split

---

## 📸 How to Use

1. Click "Browse files" to upload a digit image (ideally 28x28 or square).
2. The model will predict and display the digit.
3. You’ll see the processed image used for prediction.

---

## 🧾 Installation (Local Setup)

```bash
# Clone the repo
git clone https://github.com/yourusername/streamlit-digit-recognition.git
cd streamlit-digit-recognition
```

# Install dependencies

```bash
pip install -r requirements.txt
```

# Run the app

```bash
streamlit run app.py
```
