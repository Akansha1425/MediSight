import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="MediSight AI", layout="wide")

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🩺 MediSight AI")
st.sidebar.metric("Model Accuracy", "84%")
st.sidebar.info("AI-based Chest X-ray Diagnosis")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 3)
    model.load_state_dict(torch.load("MediSight_DenseNet_CXR_Model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# =========================
# CONFIG
# =========================
class_names = ["Normal", "COVID", "Pneumonia"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# =========================
# PREDICT FUNCTION
# =========================
def predict(image):
    img = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    return pred.item(), confidence.item(), probs.numpy()

# =========================
# GRAD-CAM FUNCTION 🔥
# =========================
def generate_gradcam(image):
    model.eval()
    
    img = transform(image).unsqueeze(0)
    img.requires_grad = True

    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Hook last conv layer
    target_layer = model.features[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(img)
    pred_class = output.argmax()

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0]
    fmap = features[0]

    weights = torch.mean(grads, dim=(2,3), keepdim=True)
    cam = torch.sum(weights * fmap, dim=1).squeeze()

    cam = torch.relu(cam)
    cam = cam / cam.max()

    cam = cam.detach().numpy()
    cam = cv2.resize(cam, (224,224))

    img_np = np.array(image.resize((224,224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    return overlay

# =========================
# UI
# =========================
st.title("🩺 MediSight AI - Advanced Dashboard")
st.markdown("Upload chest X-ray images for diagnosis")

# MULTI IMAGE UPLOAD
uploaded_files = st.file_uploader("Upload Images", type=["jpg","png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image, caption="Original", use_container_width=True)

        if st.button(f"Analyze {uploaded_file.name}"):
            pred, confidence, probs = predict(image)

            with col2:
                st.subheader("Prediction")

                if pred == 1:
                    st.error("⚠️ COVID Detected")
                elif pred == 2:
                    st.warning("⚠️ Pneumonia Detected")
                else:
                    st.success("✅ Normal")

                st.metric("Confidence", f"{confidence*100:.2f}%")

                # Probabilities
                for i, cls in enumerate(class_names):
                    st.write(cls)
                    st.progress(float(probs[0][i]))

            # GRAD-CAM
            # GRAD-CAM
            with col3:
                heatmap = generate_gradcam(image)
                st.image(heatmap, caption="🔥 AI Attention Heatmap", use_container_width=True)
                st.markdown("### 🧠 Heatmap Explanation")
                st.info("""
                        🔴 **Red / Yellow Areas** → Regions where the AI is focusing more  
                        🟢 **Green Areas** → Moderate attention  
                        🔵 **Blue Areas** → Low or no attention  
                        The highlighted regions indicate **important lung areas** that influenced the AI’s prediction.
                        """)
                st.warning("""
                           ⚠️ Note: This is an AI-assisted visualization and should be interpreted by medical professionals.
                           """)
         
           

st.markdown("---")
st.markdown("🔬 AI-powered medical assistant")