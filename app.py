import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
import base64

# Image processing functions 
def smooth_image(image):
    return cv2.blur(image, (5, 5))

def sharpen_image(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def edge_detection(image):
    return cv2.Canny(image, 100, 200)

def gradient_image(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    return np.uint8(grad_mag)

def grayscale_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def compress_image(image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, enc_img = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(enc_img, 1)

def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def median_blur(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

def kmeans_segmentation(image, k=2):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    return segmented_image.reshape(image.shape)

def remove_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(image, image, mask=threshold)
    
def white_to_pink(image):
    lower_white = np.array([200, 200, 200])  
    upper_white = np.array([255, 255, 255])  
    white_mask = cv2.inRange(image, lower_white, upper_white)
    pink_color = [255, 105, 180]
    image[white_mask > 0] = pink_color
    return image

def flip_image(image, direction='horizontal'):
    if direction == 'horizontal':
        flipped_image = cv2.flip(image, 1)
    elif direction == 'vertical':
        flipped_image = cv2.flip(image, 0)
    return flipped_image

# Function to download image
def download_image(image):
    _, buffer = cv2.imencode('.png', image)
    io_bytes = io.BytesIO(buffer)
    return io_bytes

# Streamlit UI
st.title("Image Processing App")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and display image
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Select function
    operation = st.selectbox(
        "Select an operation",
        ("Smooth", "Sharpen", "Rotate", "Edge Detection", "Gradient", "Grayscale", "Compress", "Gaussian Blur", 
         "Median Blur", "KMeans Segmentation", "Remove Background", "White to Pink", "Flip")
    )
    
    # Dynamic Parameters
    if operation in ["Gaussian Blur", "Median Blur"]:
        kernel_size = st.slider("Select kernel size", 3, 21, 5, 2)  # Kernel size for blur functions

    if operation == "Rotate":
        angle = st.slider("Select rotation angle", -180, 180, 0)
    
    if operation == "KMeans Segmentation":
        k = st.slider("Select number of segments", 2, 10, 2)
        
    if operation == "Flip":
        direction = st.selectbox("Select flip direction", ["horizontal", "vertical"])

    # Apply selected function
    if operation == "Smooth":
        result = smooth_image(image)
    elif operation == "Sharpen":
        result = sharpen_image(image)
    elif operation == "Rotate":
        result = rotate_image(image, angle)
    elif operation == "Edge Detection":
        result = edge_detection(image)
    elif operation == "Gradient":
        result = gradient_image(image)
    elif operation == "Grayscale":
        result = grayscale_image(image)
    elif operation == "Compress":
        result = compress_image(image)
    elif operation == "Gaussian Blur":
        result = gaussian_blur(image, kernel_size)
    elif operation == "Median Blur":
        result = median_blur(image, kernel_size)
    elif operation == "KMeans Segmentation":
        result = kmeans_segmentation(image, k)
    elif operation == "Remove Background":
        result = remove_background(image)
    elif operation == "White to Pink":
        result = white_to_pink(image)
    elif operation == "Flip":
        result = flip_image(image, direction)

    # Display original and processed image
    st.image(image, caption="Original Image", use_column_width=True)
    st.image(result, caption="Processed Image", use_column_width=True)

    # Provide a download button
    download_button = st.button("Download Processed Image")
    if download_button:
        result_bytes = download_image(result)
        st.download_button(
            label="Download Image",
            data=result_bytes,
            file_name="processed_image.png",
            mime="image/png"
        )

    # Explanation of the selected function
    if operation == "Smooth":
        st.write("This operation applies a simple blur to the image to reduce detail and noise.")
    elif operation == "Sharpen":
        st.write("Sharpening enhances the edges and fine details of the image.")
    elif operation == "Rotate":
        st.write(f"This operation rotates the image by {angle} degrees.")
    elif operation == "Edge Detection":
        st.write("Edge detection highlights the edges within the image.")
    elif operation == "Gradient":
        st.write("This operation calculates the gradient of the image, highlighting transitions between different pixel intensities.")
    elif operation == "Grayscale":
        st.write("Converts the image into grayscale (black & white).")
    elif operation == "Compress":
        st.write("Compresses the image to reduce file size while maintaining quality.")
    elif operation == "Gaussian Blur":
        st.write(f"This operation applies a Gaussian blur with a kernel size of {kernel_size} to soften the image.")
    elif operation == "Median Blur":
        st.write(f"Applies median blur with a kernel size of {kernel_size}, which is effective for removing salt-and-pepper noise.")
    elif operation == "KMeans Segmentation":
        st.write(f"Segments the image into {k} regions using the KMeans clustering algorithm.")
    elif operation == "Remove Background":
        st.write("Removes the background by detecting and masking the foreground objects.")
    elif operation == "White to Pink":
        st.write("Replaces all white pixels in the image with pink color.")
    elif operation == "Flip":
        st.write(f"The image is flipped in the {direction} direction.")
