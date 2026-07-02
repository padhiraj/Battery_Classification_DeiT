import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
class_names = [
    "keypad-branded",
    "keypad-non-branded",
    "others",
    "polymer-branded",
    "polymer-non-branded"
]

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# Load model
@st.cache_resource
def load_model():

    model = timm.create_model(
        "deit_base_patch16_224",
        pretrained=False,
        num_classes=5
    )

    checkpoint = torch.load(
        "best_deit_base.pth",
        map_location=device
    )

    # If you saved a checkpoint dictionary
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # If you saved only the state_dict
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model

model = load_model()

st.title("Battery Classification using DeiT-Base")
st.write("Upload a battery image to classify it.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():

        outputs = model(input_tensor)

        probs = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probs, dim=1)

    st.success(f"Prediction: {class_names[prediction.item()]}")
    st.write(f"Confidence: {confidence.item()*100:.2f}%")

    st.subheader("Class Probabilities")

    for cls, prob in zip(class_names, probs[0]):
        st.write(f"{cls}: {prob.item()*100:.2f}%")
