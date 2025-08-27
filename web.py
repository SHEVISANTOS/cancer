# app_streamlit.py
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

# -------------------------------
# Helper function to load the model
# -------------------------------
def load_model(checkpoint_path="best_model.pth"):
    """
    Load the CNN model from a checkpoint if it exists.
    Returns the model and class names.
    """
    # Define a dummy CNNModel class (replace with your actual model)
    class CNNModel(torch.nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, 1)
            self.fc = torch.nn.Linear(16*62*62, num_classes)  # Adjust based on input size

        def forward(self, x):
            x = self.conv(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    class_names = ['Normal', 'Cancer']  # Default classes
    model = CNNModel(num_classes=len(class_names))

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        if 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
    else:
        st.warning(f"Checkpoint '{checkpoint_path}' not found. Using untrained model.")

    model.eval()
    return model, class_names

# -------------------------------
# Helper function to predict image
# -------------------------------
def predict_image(model, class_names, image):
    """
    Predict the class of a PIL image using the CNN model.
    Returns the predicted label and confidence.
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Adjust based on your model
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, idx = torch.max(probs, 1)
        label = class_names[idx.item()]

    return label, confidence.item()

# -------------------------------
# Streamlit Web App UI
# -------------------------------
st.title("Lung Cancer Image Classifier")
st.write("Upload a chest scan image and the model will predict whether it shows normal tissue or cancer.")

# Load model
model, class_names = load_model()

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
   # Display uploaded image
    st.image(image, caption='Uploaded Image', use_container_width=True)


    # Predict
    label, confidence = predict_image(model, class_names, image)
    st.success(f"Prediction: **{label}** with confidence {confidence:.2f}")

st.write("BUILD BY SHEVI SANTOS")
