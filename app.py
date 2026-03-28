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

#----------------
def generate_interpretation(pred, confidence, abnormal_flag, score, boxes):
    
    interpretation = ""

    # Case 1: Normal but abnormal regions
    if pred == 0 and abnormal_flag:
        interpretation = (
            "The model predicts a normal condition, but some regions show unusual patterns. "
            "This could be due to noise or early-stage abnormalities. Further review is recommended."
        )

    # Case 2: Pneumonia / COVID detected
    elif pred != 0:
        if score > 0.3:
            interpretation = (
                "The highlighted regions show strong patterns associated with lung infection. "
                "These areas may indicate inflammation or fluid accumulation."
            )
        else:
            interpretation = (
                "The model detected signs of disease, but the affected regions are not very strong. "
                "This could indicate mild or early-stage infection."
            )

    # Case 3: Normal + no abnormality
    else:
        interpretation = (
            "No significant abnormal patterns detected. The lungs appear clear in the analyzed image."
        )

    return interpretation



# =========================
# PREDICTION
# =========================
def predict(image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    return pred.item(), confidence.item(), probs.numpy()

# =========================
# GRAD-CAM + LOCALIZATION
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

    # ===== REGION EXTRACTION =====
    heatmap_uint8 = np.uint8(255 * cam)
    _, binary = cv2.threshold(heatmap_uint8, 120, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 500:
            boxes.append((x, y, w, h))

    # ===== OVERLAY =====
    img_np = np.array(image.resize((224,224)))
    heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    for (x, y, w, h) in boxes:
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0,255,0), 2)

    return cam, overlay, boxes

# =========================
# ABNORMALITY
# =========================
def get_abnormality(cam, boxes):
    score = np.mean(cam)

    abnormal_flag = False
    if score > 0.2 or len(boxes) > 0:
        abnormal_flag = True

    return abnormal_flag, score

# =========================
# FINAL DECISION
# =========================
def final_decision(pred, confidence, abnormal_flag):
    if pred == 0 and abnormal_flag:
        return "⚠️ Review Required"

    if pred != 0 and confidence < 0.6:
        return "⚠️ Low Confidence - Verify"

    return "✅ Reliable Prediction"

# =========================
# UI
# =========================
st.title("🩺 MediSight AI - Advanced Dashboard")
st.markdown("Upload chest X-ray images for diagnosis")

uploaded_files = st.file_uploader("Upload Images", type=["jpg","png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2, col3 = st.columns(3)

        # ORIGINAL
        with col1:
            st.image(image, caption="Original", width=300)

        if st.button(f"Analyze {uploaded_file.name}"):

            pred, confidence, probs = predict(image)

            # PREDICTION PANEL
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

            # GRAD-CAM + ANALYSIS
            with col3:
                cam, heatmap, boxes = generate_gradcam(image)
                abnormal_flag, score = get_abnormality(cam, boxes)

                st.image(heatmap, caption="🔥 AI Localization (Pseudo)", width=300)

                st.markdown("### 🧠 Abnormality Analysis")

                if abnormal_flag:
                    st.error("⚠️ Abnormal Regions Detected")
                else:
                    st.success("✅ No Significant Abnormality")

                st.metric("Abnormality Score", f"{score:.3f}")
                st.write(f"Detected Regions: {len(boxes)}")

                decision = final_decision(pred, confidence, abnormal_flag)
                st.info(f"Final Decision: {decision}")
                interpretation = generate_interpretation(pred, confidence, abnormal_flag, score, boxes)
                st.markdown("### 📊 Interpretation")

                st.write(interpretation)
                st.warning("⚠️ AI assistance only. Not a medical diagnosis.")

st.markdown("---")
st.markdown("🔬 AI-powered medical assistant")