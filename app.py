import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image
import cv2

st.set_page_config(page_title="Digit Recognition App", layout="centered")

st.title("🧠 Handwritten Digit Recognition")
st.write("Upload an image (28x28 grayscale) of a digit (0-9) to predict it.")

# Load and train the model
@st.cache_resource
def train_model():
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(gamma=0.001)
    model.fit(X_train, y_train)
    return model, digits

model, digits = train_model()

# File uploader
uploaded_file = st.file_uploader("Upload your digit image", type=["png", "jpg", "jpeg"])

def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((8, 8), Image.LANCZOS)
    img_array = np.array(img)
    img_array = 16 - (img_array / 16)  # Normalize to match digits dataset (0–16 scale)
    flat = img_array.flatten()
    return flat

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=False, width=200)

    with st.spinner('Predicting...'):
        processed = preprocess_image(image)
        prediction = model.predict([processed])[0]
        st.success(f"✅ Predicted Digit: **{prediction}**")

        # Show processed image
        st.subheader("Processed Image (8x8 Grayscale)")
        fig, ax = plt.subplots()
        ax.imshow(16 - processed.reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
        st.pyplot(fig)

st.markdown("---")
st.markdown("**Note:** This model uses the `sklearn` digits dataset. For MNIST or real-world use, consider a CNN with TensorFlow or PyTorch.")
