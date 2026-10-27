"""Interactive MNIST classification and generation lab."""

from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

from models import MNISTClassifier, MNISTVAE


ROOT = Path(__file__).resolve().parent
CLASSIFIER_PATH = ROOT / "models" / "classifier.pt"
VAE_PATH = ROOT / "models" / "vae.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Generative vs. Discriminative Model Lab", layout="wide")

st.markdown("""
<style>
:root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    --text-color: #1f2937 !important;
    --secondary-text-color: #344054 !important;
    --heading-color: #172554 !important;
    --primary-color: #dc2626 !important;
    color: #1f2937 !important;
}
html, body, html body .stApp, html body [data-testid="stAppViewContainer"] {
    color: #1f2937 !important;
}
.stApp {background: radial-gradient(ellipse at top left, #eef4ff 0, #f8f9fc 42%, #f4f2ff 100%);}
.block-container {max-width: 1180px; padding: 2.2rem 1.8rem 3rem;}
[data-testid="stVerticalBlock"] {gap: 1rem;}
[data-testid="stHorizontalBlock"] {gap: 1.5rem; align-items: stretch;}
html body .stApp p, html body .stApp li,
html body .stApp [data-testid="stMarkdownContainer"] p,
html body .stApp [data-testid="stMarkdownContainer"] li,
html body .stApp [data-testid="stMarkdownContainer"] blockquote,
html body .stApp [data-testid="stMarkdownContainer"] code,
html body .stApp [data-testid="stCaptionContainer"],
html body .stApp [data-testid="stCaptionContainer"] *,
html body .stApp .stCaption, html body .stApp [data-testid="stText"],
html body .stApp [data-testid="stWidgetLabel"],
html body .stApp [data-testid="stWidgetLabel"] *,
html body .stApp label, html body .stApp label *,
html body .stApp [data-testid="stMetricLabel"],
html body .stApp [data-testid="stMetricLabel"] *,
html body .stApp [data-testid="stMetricValue"],
html body .stApp [data-testid="stMetricValue"] *,
html body .stApp [data-testid="stAlert"] *,
html body .stApp [data-testid="stRadio"] *,
html body .stApp [data-testid="stSlider"] *,
html body .stApp [data-baseweb="select"] *,
html body .stApp input, html body .stApp textarea {
    color: #263248 !important;
    opacity: 1 !important;
}
html body .stApp input::placeholder, html body .stApp textarea::placeholder {
    color: #667085 !important;
    opacity: 1 !important;
}
/* Place heading colors after body text rules so theme/Markdown styles cannot wash them out. */
html body .stApp h1, html body .stApp h2, html body .stApp h3,
html body .stApp h4, html body .stApp h5, html body .stApp h6,
html body .stApp [data-testid="stMarkdownContainer"] h1,
html body .stApp [data-testid="stMarkdownContainer"] h2,
html body .stApp [data-testid="stMarkdownContainer"] h3,
html body .stApp [data-testid="stMarkdownContainer"] h4,
html body .stApp [data-testid="stMarkdownContainer"] h5,
html body .stApp [data-testid="stMarkdownContainer"] h6,
html body [data-testid="stMain"] [data-testid="stMarkdownContainer"] h1,
html body [data-testid="stMain"] [data-testid="stMarkdownContainer"] h2,
html body [data-testid="stMain"] [data-testid="stMarkdownContainer"] h3,
html body [data-testid="stMain"] [data-testid="stMarkdownContainer"] h4,
html body [data-testid="stMain"] [data-testid="stMarkdownContainer"] h5,
html body [data-testid="stMain"] [data-testid="stMarkdownContainer"] h6 {
    color: #172554 !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    text-shadow: none !important;
}
.hero {padding: 1.6rem 1.9rem; border-radius: 20px; color: white; background: linear-gradient(120deg,#243b80,#7054c7); margin-bottom: 1.3rem; box-shadow: 0 12px 30px rgba(36,59,128,.16);}
html body .stApp .hero, html body .stApp .hero * {color: #fff !important; opacity: 1 !important;}
.hero h1 {margin: 0; font-size: 2.2rem; line-height: 1.2;}
.hero p {margin: .65rem 0 0; opacity: .94; font-size: 1.05rem;}
[data-testid="stButton"] button {background: #dc2626 !important; color: #fff !important; border: 1px solid #b91c1c !important; border-radius: 10px; font-weight: 650; min-height: 2.7rem; transition: background .15s ease, transform .15s ease;}
[data-testid="stButton"] button:hover:not(:disabled) {background: #b91c1c !important; color: #fff !important; border-color: #991b1b !important; transform: translateY(-1px);}
[data-testid="stButton"] button:disabled {background: #e5e7eb !important; color: #6b7280 !important; border-color: #d1d5db !important; cursor: not-allowed;}
[data-testid="stMetric"], [data-testid="stAlert"] {border-radius: 12px;}
@media (max-width: 760px) {
    .block-container {padding: 1.2rem 1rem 2rem;}
    [data-testid="stHorizontalBlock"] {gap: .8rem;}
    .hero {padding: 1.3rem 1.2rem;}
    .hero h1 {font-size: 1.7rem;}
}
</style>
<div class="hero"><h1>Generative vs. Discriminative Model Lab</h1>
<p>Draw a digit, see how a classifier reasons, then sample new handwriting from a trained VAE.</p></div>
""", unsafe_allow_html=True)


def model_file_stamp(path: Path) -> tuple[int, int] | None:
    """Return a cache key that changes when a model file is created or updated."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


@st.cache_resource(show_spinner=False)
def load_models(classifier_stamp, vae_stamp):
    # The stamps are cache inputs so newly trained weights trigger a reload.
    _ = (classifier_stamp, vae_stamp)
    classifier = vae = None
    errors = []
    if CLASSIFIER_PATH.exists():
        try:
            classifier = MNISTClassifier().to(DEVICE)
            classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE, weights_only=True))
            classifier.eval()
        except Exception as exc:
            classifier = None
            errors.append(f"Classifier could not be loaded: {exc}")
    else:
        errors.append("Classifier weights are missing.")
    if VAE_PATH.exists():
        try:
            vae = MNISTVAE().to(DEVICE)
            vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE, weights_only=True))
            vae.eval()
        except Exception as exc:
            vae = None
            errors.append(f"VAE could not be loaded: {exc}")
    else:
        errors.append("VAE weights are missing.")
    return classifier, vae, errors


def grayscale_on_white(image: Image.Image) -> Image.Image:
    """Orient an image, composite transparency onto white, and return grayscale."""
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("L")
    return ImageOps.grayscale(image)


def mnist_ink_array(image: Image.Image) -> np.ndarray:
    """Return uint8 pixels with the digit bright and the background dark."""
    arr = np.asarray(grayscale_on_white(image), dtype=np.uint8).copy()
    # Convert white paper with dark ink to MNIST convention; preserve white ink on black.
    if float(np.median(arr)) > 127:
        arr = 255 - arr
    return arr


def image_has_ink(image: Image.Image) -> bool:
    """Reject blank canvases/uploads after using the classifier's polarity convention."""
    return bool(np.any(mnist_ink_array(image) > 24))


def canvas_data_to_image(image_data: np.ndarray) -> Image.Image | None:
    """Convert drawable-canvas RGBA/RGB/grayscale output into a PIL image."""
    pixels = np.asarray(image_data)
    if pixels.ndim == 3 and pixels.shape[2] == 1:
        pixels = pixels[:, :, 0]
    if pixels.ndim not in (2, 3):
        return None
    if pixels.ndim == 3 and pixels.shape[2] not in (3, 4):
        return None
    if np.issubdtype(pixels.dtype, np.floating) and pixels.size and pixels.max() <= 1:
        pixels = pixels * 255
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def prepare_image(image: Image.Image) -> tuple[torch.Tensor, Image.Image]:
    """Crop digit ink and build the normalized single-channel 1x1x28x28 input."""
    arr = mnist_ink_array(image)
    ink = arr > 24
    if ink.any():
        ys, xs = np.where(ink)
        pad = max(2, int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.12))
        left, right = max(0, xs.min() - pad), min(arr.shape[1], xs.max() + pad + 1)
        top, bottom = max(0, ys.min() - pad), min(arr.shape[0], ys.max() + pad + 1)
        arr = arr[top:bottom, left:right]
    digit = Image.fromarray(arr).resize((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
    tensor = torch.from_numpy(np.asarray(canvas, dtype=np.float32).copy() / 255.0)[None, None]
    return tensor.to(DEVICE), canvas


classifier, vae, model_errors = load_models(
    model_file_stamp(CLASSIFIER_PATH), model_file_stamp(VAE_PATH)
)
if model_errors:
    st.warning("Some trained weights are not available yet. Train them from the project terminal with `python train_models.py`. The app will still open, and each feature will explain when its model is missing.")
    with st.expander("Model status"):
        for error in model_errors:
            st.write(f"- {error}")

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.subheader("1. Draw or upload a digit")
    st.caption("Use a single centered digit. The drawing pad uses black ink on a white background.")
    mode = st.radio("Input method", ["Draw", "Upload"], horizontal=True, label_visibility="collapsed")
    input_image = None
    if mode == "Draw":
        canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)", stroke_width=14, stroke_color="#111827",
            background_color="#ffffff", height=280, width=280, drawing_mode="freedraw",
            key="digit_canvas",
        )
        if canvas.image_data is not None:
            canvas_image = canvas_data_to_image(canvas.image_data)
            if canvas_image is not None and image_has_ink(canvas_image):
                input_image = canvas_image
    else:
        uploaded = st.file_uploader("Choose a digit image", type=["png", "jpg", "jpeg", "bmp", "webp"])
        if uploaded is not None:
            try:
                with Image.open(uploaded) as source:
                    uploaded_image = ImageOps.exif_transpose(source).copy()
                if image_has_ink(uploaded_image):
                    input_image = uploaded_image
                    st.image(input_image, caption="Uploaded image", width=220)
                else:
                    st.info("That image appears blank. Try an image with a visible digit.")
            except (OSError, ValueError) as exc:
                st.error(f"That image could not be opened. Try a valid PNG or JPEG. ({exc})")
    predict = st.button("Predict digit", type="primary", use_container_width=True)

with right:
    st.subheader("2. Classifier result")
    if predict:
        if input_image is None:
            st.info("Draw a visible digit or upload an image with a visible digit first.")
        elif classifier is None:
            st.error("Classification is unavailable until `models/classifier.pt` is trained. Run `python train_models.py` in the project terminal.")
        else:
            tensor, preview = prepare_image(input_image)
            classifier.eval()
            with torch.no_grad():
                probabilities = torch.softmax(classifier(tensor), dim=1)[0].cpu().numpy()
            pred = int(probabilities.argmax())
            st.image(preview.resize((168, 168), Image.Resampling.NEAREST), caption="Normalized 28 x 28 model input", width=168)
            st.metric("Predicted digit", str(pred), f"{probabilities[pred] * 100:.1f}% confidence")
            st.caption("Confidence is the classifier's softmax score, not a guarantee of correctness.")
            st.bar_chart({str(i): float(probabilities[i]) for i in range(10)}, y_label="Probability", x_label="Digit")
    else:
        st.info("Your prediction and the ten-class probability chart will appear here.")

st.divider()
st.subheader("3. Generate new handwritten digits")
st.write("The VAE samples latent patterns learned from MNIST and decodes them into new 28 x 28 images.")
count = st.slider("Number of samples", min_value=4, max_value=16, value=8, step=4)
generate = st.button("Generate new digits", type="primary", disabled=vae is None, use_container_width=False)
if generate:
    if vae is None:
        st.error("Generation is unavailable until `models/vae.pt` is trained. Run `python train_models.py` in the project terminal.")
    else:
        with torch.inference_mode():
            samples = vae.decode(torch.randn(count, vae.latent_dim, device=DEVICE)).cpu().squeeze(1).numpy()
        cols = st.columns(min(8, count))
        for i, sample in enumerate(samples):
            with cols[i % len(cols)]:
                st.image(sample, clamp=True, width=100, caption=f"Sample {i + 1}")

st.divider()
st.subheader("Classification vs. generation")
col_a, col_b = st.columns(2, gap="large")
with col_a:
    st.markdown("#### Discriminative model")
    st.write("A discriminative model learns to distinguish labels from inputs. Here the CNN receives your digit and predicts which class (0-9) best matches it.")
with col_b:
    st.markdown("#### Generative model")
    st.write("A generative model learns patterns in examples. Here the VAE learns a latent representation of MNIST digits and decodes sampled points into new digit-like images.")
st.caption("The probability chart is a useful visualization of how the classifier distributes its belief across all ten digit classes.")
