import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import os
import gdown

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Battery Classification using DeiT-Base",
    page_icon="🔋",
    layout="centered"
)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Google Drive Model
# -----------------------------
MODEL_PATH = "best_deit_base.pth"

MODEL_URL = "https://drive.google.com/uc?id=1O3af0uBFtr3oZMnWQyivK8iho9bZ7CNT"

# Download model only once
if not os.path.exists(MODEL_PATH):

    with st.spinner("Downloading trained model... This may take a few minutes."):

        gdown.download(
            MODEL_URL,
            MODEL_PATH,
            quiet=False
        )

# -----------------------------
# Class Names
# -----------------------------
class_names = [
    "keypad-branded",
    "keypad-non-branded",
    "others",
    "polymer-branded",
    "polymer-non-branded"
]

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

    model = timm.create_model(
        "deit_base_patch16_224",
        pretrained=False,
        num_classes=5
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    # If checkpoint dictionary
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.to(device)

    model.eval()

    return model

model = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔋 Battery Classification using DeiT-Base")

st.write(
    "Upload a battery image and the model will predict its category."
)

uploaded_file = st.file_uploader(
    "Upload Battery Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=350
    )

    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    with torch.inference_mode():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    st.success(
        f"Prediction : {class_names[prediction.item()]}"
    )

    st.write(
        f"Confidence : {confidence.item()*100:.2f}%"
    )

    st.subheader("Class Probabilities")

    for cls, prob in zip(class_names, probabilities[0]):

        st.write(
            f"{cls}: {prob.item()*100:.2f}%"
        )
