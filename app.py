import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="DL Team - Task 3", layout="centered")
st.title("Image Processing with Original + Transformed View")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    # Open image safely
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # Convert uploaded file to OpenCV format
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # --- Resize controls ---
    st.subheader("Resize Controls")
    col1, col2 = st.columns(2)
    with col1:
        target_w = st.slider("Width", min_value=50, max_value=img_bgr.shape[1], value=1080)
    with col2:
        target_h = st.slider("Height", min_value=50, max_value=img_bgr.shape[0], value=720)

    resized = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

    if st.button("Resize to 1080x720"):
        resized = cv2.resize(img_bgr, (1080, 720), interpolation=cv2.INTER_AREA)

    # --- Processing controls ---
    st.subheader("Processing Options")
    gray_brightness = st.slider("Gray brightness", -100, 100, 0, step=5)
    gray_contrast = st.slider("Gray contrast", 0.5, 3.0, 1.0, step=0.1)
    blur_kernel = st.slider("Blur kernel size (odd)", 1, 51, 7, step=2)
    blur_sigma = st.slider("Gaussian sigma", 0.0, 10.0, 1.5, step=0.1)
    canny_low = st.slider("Canny lower threshold", 0, 255, 50)
    canny_high = st.slider("Canny upper threshold", 0, 255, 150)

    effects = st.multiselect(
        "Select effects to apply (in order)",
        ["Gray", "Blur", "Edge"],
        default=[]
    )

    # --- Display images side by side ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), caption="Original", width=400)

    with col2:
        disp = resized.copy()
        caption = "Processed Image"

        for effect in effects:
            if effect == "Gray":
                gray = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
                gray = cv2.convertScaleAbs(gray, alpha=gray_contrast, beta=gray_brightness)
                disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                caption = f"Gray (contrast={gray_contrast}, brightness={gray_brightness})"

            elif effect == "Blur":
                k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
                if k >= min(disp.shape[:2]):
                    st.warning("Kernel size too large for this image. Try smaller value.")
                else:
                    disp = cv2.GaussianBlur(disp, (k, k), blur_sigma)
                    caption = f"Blurred (kernel={k}, sigma={blur_sigma})"

            elif effect == "Edge":
                gray = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, canny_low, canny_high)
                disp = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                caption = f"Edges (low={canny_low}, high={canny_high})"

        st.subheader("Processed Image")
        st.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), caption=caption, width=400)

else:
    st.warning("Please upload an image file.")
