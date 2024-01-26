import numpy as np 
import cv2
from PIL import Image
import io
import streamlit as st

def colorize_image(input_bytes):
    # Use PIL to load the image
    input_image = Image.open(io.BytesIO(input_bytes)).convert("RGB")
    input_image = np.array(input_image)

    net = cv2.dnn.readNetFromCaffe('colorization_deploy_v2.prototxt', 'colorization_release_v2.caffemodel')
    pts = np.load('pts_in_hull.npy')

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)

    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]

    scaled = input_image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (input_image.shape[1], input_image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")

    return colorized

def main():
    st.title("Image Colorizer")
    st.markdown("<p style='margin-top:-15px; margin-bottom:5px; font-size: 10px;'>by PAVAN SHELKE</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a black and white image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the original image
        original_image_bytes = uploaded_file.read()
        original_image = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
        st.image(original_image, caption="Original Image", use_column_width=True)

        # Colorize the image
        if st.button("Colorize"):
            colorized_image = colorize_image(original_image_bytes)

            # Display the colorized image
            if colorized_image is not None:
                st.image(colorized_image, caption="Colorized Image", use_column_width=True)

if __name__ == "__main__":
    main()
