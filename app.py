import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt

st.set_page_config(page_title="DigitLab | Explainable AI", page_icon="✍️", layout="wide")

st.markdown("""
<style>
.main {background: linear-gradient(180deg,#f7f8ff 0%,#ffffff 55%);}
.block-container {padding-top: 1.5rem; max-width: 1200px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 22px; color: white;
background: linear-gradient(115deg,#111827,#4338ca 65%,#7c3aed); margin-bottom:1rem;}
.hero h1 {font-size:2.2rem; margin:0;}
.hero p {opacity:.9; margin:.5rem 0 0;}
</style>
<div class="hero"><h1>✍️ DigitLab: Handwritten Digit Recognition</h1>
<p>Compare four neural-network algorithms and inspect what the model predicts.</p></div>
""", unsafe_allow_html=True)

MODEL_FILES = {
    "MLP (Dense Neural Network)": "models/MLP.keras",
    "LeNet-5-style CNN": "models/LeNet5.keras",
    "Custom CNN": "models/CustomCNN.keras",
    "Small ResNet": "models/ResNet.keras",
}
st.caption("Train the four models first using `python train_models.py`. This app loads saved models and does not train on each visit.")

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

def prepare_image(image):
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    # Make digit white on black, matching MNIST convention.
    a = np.array(image)
    if a.mean() > 127:
        image = ImageOps.invert(image)
    image = image.point(lambda p: 255 if p > 55 else 0)
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    if image.width == 0 or image.height == 0:
        return np.zeros((28,28,1), dtype=np.float32)
    image.thumbnail((20,20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28,28), 0)
    canvas.paste(image, ((28-image.width)//2, (28-image.height)//2))
    arr = np.asarray(canvas).astype("float32") / 255.0
    return arr[..., None]

def gradcam(model, x):
    conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    if not conv_layers:
        return None
    target = conv_layers[-1]
    try:
        grad_model = tf.keras.Model(model.inputs, [target.output, model.output])
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(x, training=False)
            class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]
        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return None
        weights = tf.reduce_mean(grads, axis=(0,1,2))
        cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
        cam = tf.maximum(cam, 0)
        mx = tf.reduce_max(cam)
        if float(mx) > 0:
            cam = cam / mx
        return cam.numpy()
    except Exception:
        return None

left, right = st.columns([1,1], gap="large")
with left:
    st.subheader("1 · Choose a model")
    chosen = st.selectbox("Algorithm", list(MODEL_FILES.keys()))
    model_path = MODEL_FILES[chosen]
    st.subheader("2 · Provide a digit")
    source = st.radio("Input method", ["Upload image", "Draw digit"], horizontal=True)
    uploaded = None
    if source == "Upload image":
        uploaded = st.file_uploader("Upload a handwritten digit image", type=["png","jpg","jpeg"])
        if uploaded:
            original = Image.open(uploaded).convert("RGB")
            st.image(original, caption="Uploaded image", width=220)
    else:
        try:
            from streamlit_drawable_canvas import st_canvas
            canvas_result = st_canvas(
                fill_color="rgba(0,0,0,0)", stroke_width=16, stroke_color="#FFFFFF",
                background_color="#000000", height=280, width=280, drawing_mode="freedraw",
                key="digit_canvas", update_streamlit=True, return_image_data=True
            )
            if canvas_result is not None and canvas_result.image_data is not None:
                canvas_arr = np.asarray(canvas_result.image_data).astype(np.uint8)
                if canvas_arr.ndim == 3 and np.max(canvas_arr[:,:,:3]) > 0:
                    uploaded = Image.fromarray(canvas_arr, mode="RGBA").convert("RGB")
        except Exception as e:
            st.warning("Drawing canvas is unavailable. Use image upload instead.")
            st.caption(str(e))

    predict = st.button("✨ Predict digit", type="primary", use_container_width=True)
with right:
    st.subheader("3 · Prediction and explanation")
    if predict:
        if not os.path.exists(model_path):
            st.error(f"Model file not found: `{model_path}`. Run `python train_models.py` first.")
        elif uploaded is None:
            st.info("Draw a digit or upload an image first.")
        else:
            try:
                model = load_model(model_path)
                processed = prepare_image(uploaded)
                x = processed[None, ...]
                probs = model.predict(x, verbose=0)[0]
                pred = int(np.argmax(probs))
                confidence = float(probs[pred])
                st.markdown(f"### Predicted digit: **{pred}**")
                st.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")
                c1, c2 = st.columns(2)
                with c1:
                    st.image((processed[:,:,0]*255).astype("uint8"), caption="Preprocessed 28×28 input", width=180)
                with c2:
                    st.metric("Top prediction", str(pred), f"{confidence*100:.1f}% confidence")
                fig, ax = plt.subplots(figsize=(7,2.7))
                ax.bar(range(10), probs)
                ax.set_xticks(range(10)); ax.set_xlabel("Digit class"); ax.set_ylabel("Probability")
                ax.set_ylim(0,1); ax.set_title("Probability for each digit")
                st.pyplot(fig, clear_figure=True)
                cam = gradcam(model, x)
                if cam is not None:
                    st.markdown("**Grad-CAM explanation**")
                    fig2, ax2 = plt.subplots(figsize=(3.5,3.5))
                    ax2.imshow(processed[:,:,0], cmap="gray")
                    ax2.imshow(tf.image.resize(cam[...,None], (28,28)).numpy().squeeze(), cmap="jet", alpha=.42)
                    ax2.axis("off"); ax2.set_title("Highlighted model focus")
                    st.pyplot(fig2, clear_figure=True)
                else:
                    st.info("Grad-CAM is not available for this architecture; the prediction and class probabilities are still shown.")
            except Exception as e:
                st.error("Prediction failed. Check the uploaded image and model file.")
                st.exception(e)
    else:
        st.info("Choose an algorithm, provide a digit, then press **Predict digit**.")
        st.markdown("**Included algorithms**")
        st.markdown("- MLP — dense neural network\n- LeNet-5-style CNN — classic convolutional model\n- Custom CNN — deeper convolutional feature extractor\n- Small ResNet — residual connections")

st.divider()
st.caption("Dataset: MNIST handwritten digits (0–9). Use the evaluation files generated by training for your report; do not invent metric values.")
