"""Ứng dụng web xử lý ảnh X-quang phổi (Streamlit).

Chạy từ repo root:
    streamlit run app.py
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import image_processing as ip
from utils import CLASSES, list_dataset, load_gray

BASIC_GROUP = "Xử lý cơ bản"
ENHANCE_GROUP = "Tăng cường ảnh"

BASIC_TECHNIQUES = ["Xoay", "Lật", "Cắt", "Khử nhiễu", "Tách biên", "Trừ nền"]
ENHANCE_TECHNIQUES = ["Histogram Eq", "CLAHE", "Gamma", "Unsharp mask", "Pipeline đề xuất"]


def load_uploaded_gray(uploaded_file):
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.error("Không đọc được ảnh, hãy chọn file PNG/JPG/JPEG/BMP.")
        st.stop()
    return img

def build_histogram_figure(img_a, img_b, label_a, label_b):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, im, label in ((axes[0], img_a, label_a), (axes[1], img_b, label_b)):
        ax.hist(im.ravel(), bins=256, range=(0, 255), color="steelblue")
        ax.set_title(label, fontsize=10)
        ax.set_xlim(0, 255)
        ax.set_xlabel("Cường độ pixel")
        ax.set_ylabel("Số pixel")
    fig.tight_layout()
    return fig

def get_side_panel_params(group, technique):
    params = {}
    if group == BASIC_GROUP:
        if technique == "Xoay":
            params["angle"] = st.slider("Góc xoay (độ)", -180, 180, 45)
        elif technique == "Lật":
            params["mode"] = st.selectbox("Kiểu lật", ["horizontal", "vertical", "both"])
        elif technique == "Cắt":
            params["margin"] = st.slider("Margin (px)", 0, 100, 10)
        elif technique == "Khử nhiễu":
            params["method"] = st.selectbox("Phương pháp", ["median", "gaussian", "nlm"])
            if params["method"] == "gaussian":
                params["gaussian_ksize"] = st.slider("Kernel Gaussian", 3, 15, 5, step=2)
            elif params["method"] == "median":
                params["median_ksize"] = st.slider("Kernel Median", 3, 15, 5, step=2)
            elif params["method"] == "nlm":
                params["nlm_h"] = st.slider("Cường độ khử nhiễu (h)", 1, 30, 10)
        elif technique == "Tách biên":
            params["method"] = st.selectbox("Phương pháp", ["canny", "sobel", "laplacian"])
            if params["method"] == "canny":
                params["canny_low"] = st.slider("Ngưỡng dưới", 0, 255, 50)
                params["canny_high"] = st.slider("Ngưỡng trên", 0, 255, 150)
            elif params["method"] == "sobel":
                params["sobel_ksize"] = st.select_slider("Kích thước kernel Sobel", options=[1, 3, 5, 7], value=3)
            elif params["method"] == "laplacian":
                params["laplacian_ksize"] = st.select_slider("Kích thước kernel Laplacian", options=[1, 3, 5, 7], value=3)
        elif technique == "Trừ nền":
            params["morph_kernel_size"] = st.slider(
                "Kích thước kernel morphology (khi không có mask)", 3, 21, 7, step=2
            )
    else:
        if technique == "CLAHE":
            params["clip_limit"] = st.slider("Clip limit", 0.5, 5.0, 2.0, 0.1)
            tile = st.slider("Kích thước tile", 2, 16, 8)
            params["tile_grid_size"] = (tile, tile)
        elif technique == "Gamma":
            params["gamma"] = st.slider("Gamma", 0.2, 3.0, 1.5, 0.05)
        elif technique == "Unsharp mask":
            params["sigma"] = st.slider("Sigma", 0.1, 5.0, 2.0, 0.1)
            params["amount"] = st.slider("Amount", 0.1, 3.0, 1.5, 0.1)
        elif technique == "Pipeline đề xuất":
            params["clip_limit"] = st.slider("Clip limit", 0.5, 5.0, 2.0, 0.1)
            tile = st.slider("Kích thước tile", 2, 16, 8)
            params["tile_grid_size"] = (tile, tile)
            params["median_ksize"] = st.slider("Kernel median cuối", 1, 9, 3, step=2)
    return params

def apply_technique(img, mask, group, technique, params):
    if group == BASIC_GROUP:
        if technique == "Xoay":
            return ip.rotate_image(img, params["angle"])
        if technique == "Lật":
            return ip.flip_image(img, params["mode"])
        if technique == "Cắt":
            return ip.crop_image(img, mask, margin=params["margin"])
        if technique == "Khử nhiễu":
            return ip.denoise_image(
                img, params["method"],
                gaussian_ksize=params.get("gaussian_ksize", 5),
                median_ksize=params.get("median_ksize", 5),
                nlm_h=params.get("nlm_h", 10),
            )
        if technique == "Tách biên":
            return ip.edge_detect(
                img, params["method"],
                canny_low=params.get("canny_low", 50),
                canny_high=params.get("canny_high", 150),
                sobel_ksize=params.get("sobel_ksize", 3),
                laplacian_ksize=params.get("laplacian_ksize", 3),
            )
        if technique == "Trừ nền":
            return ip.background_subtract(img, mask, morph_kernel_size=params.get("morph_kernel_size", 7))
    else:
        if technique == "Histogram Eq":
            return ip.hist_eq(img)
        if technique == "CLAHE":
            return ip.clahe_enhance(img, clip_limit=params["clip_limit"], tile_grid_size=params["tile_grid_size"])
        if technique == "Gamma":
            return ip.gamma_correction(img, params["gamma"])
        if technique == "Unsharp mask":
            return ip.unsharp_mask(img, sigma=params["sigma"], amount=params["amount"])
        if technique == "Pipeline đề xuất":
            return ip.recommended_pipeline(
                img,
                clip_limit=params.get("clip_limit", 2.0),
                tile_grid_size=params.get("tile_grid_size", (8, 8)),
                median_ksize=params.get("median_ksize", 3),
            )
    raise ValueError(f"Không rõ kỹ thuật: {technique}")

# Sidebar: nguồn ảnh

def main():
    st.set_page_config(page_title="Xử lý ảnh X-quang phổi", layout="wide")

    st.sidebar.title("Bảng điều khiển")

    source = st.sidebar.radio("Nguồn ảnh", ["Tải ảnh lên", "Ảnh mẫu từ dataset"])

    img = None
    mask = None
    img_label = ""

    if source == "Tải ảnh lên":
        uploaded = st.sidebar.file_uploader("Chọn ảnh X-quang", type=["png", "jpg", "jpeg", "bmp"])
        uploaded_mask = st.sidebar.file_uploader("Mask phổi (tùy chọn)", type=["png", "jpg", "jpeg", "bmp"])
        if uploaded is None:
            st.info("Hãy tải lên một ảnh X-quang để bắt đầu.")
            return
        img = load_uploaded_gray(uploaded)
        img_label = uploaded.name
        if uploaded_mask is not None:
            mask = load_uploaded_gray(uploaded_mask)
    else:
        class_name = st.sidebar.selectbox("Lớp", CLASSES)
        items = list_dataset([class_name])
        if not items:
            st.error("Không có ảnh trong lớp này. Chạy extract_data_kaggle.py trước.")
            return
        idx = int(st.sidebar.number_input("Ảnh thứ", min_value=0, max_value=len(items) - 1, value=0, step=1))
        cls, img_path, mask_path = items[idx]
        img = load_gray(img_path)
        mask = load_gray(mask_path) if mask_path else None
        img_label = f"{cls} / {img_path.split('/')[-1]}"

    st.sidebar.markdown("---")
    group = st.sidebar.radio("Nhóm xử lý", [BASIC_GROUP, ENHANCE_GROUP])
    techniques = BASIC_TECHNIQUES if group == BASIC_GROUP else ENHANCE_TECHNIQUES
    technique = st.sidebar.selectbox("Kỹ thuật", techniques)
    params = get_side_panel_params(group, technique)

    if technique in ("Cắt", "Trừ nền") and mask is None:
        st.sidebar.caption("Không có mask — dùng phương án dự phòng (center-crop / Otsu).")

    show_hist = st.sidebar.checkbox("Hiện histogram", value=True)

    result = apply_technique(img, mask, group, technique, params)

    st.title("Xử lý ảnh X-quang phổi")
    st.caption(f"Ảnh nguồn: **{img_label}**  |  Kích thước: {img.shape[1]}×{img.shape[0]}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh gốc")
        st.image(img, caption=f"Gốc ({img.shape[1]}×{img.shape[0]})", use_container_width=True)
    with col2:
        st.subheader("Kết quả")
        st.image(result, caption=f"{technique} ({result.shape[1]}×{result.shape[0]})", use_container_width=True)

    if show_hist:
        st.pyplot(build_histogram_figure(img, result, "Histogram gốc", f"Histogram {technique}"))

    ok, buf = cv2.imencode(".png", result)
    if ok:
        st.download_button(
            "Tải ảnh kết quả (PNG)",
            data=buf.tobytes(),
            file_name=f"result_{technique}.png",
            mime="image/png",
        )


if __name__ == "__main__":
    main()
