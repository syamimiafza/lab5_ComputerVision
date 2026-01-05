import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="CPU Image Classification (ResNet18)",
    layout="wide",
)

st.title("Computer Vision: Image Classification (CPU)")
st.write(
    """
    This app performs **image classification** using **PyTorch + Torchvision** with a
    **pre-trained ResNet18** model (ImageNet).  
    - CPU only  
    - Upload JPG/PNG  
    - Shows **Top-5 predictions** + probabilities  
    - Displays a **bar chart** of probabilities  
    """
)


device = torch.device("cpu")  # CPU only

@st.cache_resource
def load_model_and_weights():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    model.to(device)
    return model, weights

model, weights = load_model_and_weights()

preprocess = weights.transforms()
labels = weights.meta["categories"]  # ImageNet class labels


st.subheader("Step 6: Upload an Image")
uploaded_file = st.file_uploader(
    "Upload an image file (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("👆 Upload an image to start classification.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded Image", use_container_width=True)

input_tensor = preprocess(image).unsqueeze(0).to(device)  # shape [1, 3, 224, 224]

with torch.no_grad():
    logits = model(input_tensor)
    probs = F.softmax(logits[0], dim=0)

#  Top-5 predictions

top5_prob, top5_idx = torch.topk(probs, 5)

top5_labels = [labels[i] for i in top5_idx.tolist()]
top5_probs = [float(p) for p in top5_prob.tolist()]

st.subheader("Step 8: Top-5 Predicted Classes")
for i in range(5):
    st.write(f"**{i+1}. {top5_labels[i]}** — Probability: {top5_probs[i]:.4f}")

# Table (Pandas)
df = pd.DataFrame({
    "Rank": [1, 2, 3, 4, 5],
    "Class": top5_labels,
    "Probability": top5_probs
})
st.subheader("Top-5 Predictions Table")
st.dataframe(df, use_container_width=True)

# Bar chart visualization
st.subheader("Step 9: Probability Bar Chart")

fig, ax = plt.subplots()
ax.bar(top5_labels, top5_probs)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Probability")
ax.set_title("Top-5 Prediction Probabilities")
plt.xticks(rotation=25, ha="right")

st.pyplot(fig)