# Image Colorization App

This project is an image colorization application that transforms black and white images into colorized versions using deep learning techniques. It leverages a pre-trained model based on the colorization_deploy_v2 architecture using OpenCV's DNN (Deep Neural Networks) module and the Caffe framework.

## How It Works

The app uses a deep neural network, implemented using OpenCV and NumPy, to predict and add color to black and white images. The model has been trained on a diverse dataset to generate vibrant and realistic colorizations.

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run the Streamlit app: `streamlit run app.py`

## Usage

1. Upload a black and white image using the provided file uploader.
2. Click the "Colorize" button to see the colorized version of the image.

## Files

- `app.py`: Streamlit UI file for user interaction.
- `colorization_deploy_v2.prototxt`: Deploy file for the colorization model.
- `colorization_release_v2.caffemodel`: Pre-trained weights for the colorization model.
- `pts_in_hull.npy`: Data file used in the colorization model.
- `requirements.txt`: File containing project dependencies.

## Dependencies

- NumPy
- OpenCV
- Streamlit
