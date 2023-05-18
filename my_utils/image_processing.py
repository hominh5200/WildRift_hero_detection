import cv2
from skimage.feature import hog
from skimage import exposure
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_hog_descriptor(image):
    """Convert the image to grayscale"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute HOG features
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys", visualize=True)
    
    return features

def calculate_hog_similarity(image1, image2):
    """Calculate HOG similarity between 2 images"""    
    # Calculate the HOG descriptors
    hog1 = calculate_hog_descriptor(image1)
    hog2 = calculate_hog_descriptor(image2)
    
    # Calculate the similarity score using cosine similarity
    similarity_score = np.dot(hog1, hog2) / (np.linalg.norm(hog1) * np.linalg.norm(hog2))
    
    return similarity_score

def resize_image(image, image_tgt):
    """Reshape image into target image size"""
    image_tgt_h, image_tgt_w, _ = image_tgt.shape
    
    # Resize the image using OpenCV
    resized_image = cv2.resize(image, (image_tgt_w, image_tgt_h))
    
    return resized_image

def calculate_structural_similarity(image1, image2):
    """Convert images to gray scale and calculate structural similarity of them"""
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    similarity_score = ssim(image1, image2)
    return similarity_score

"""# Example usage
image1_path = 'images\\Darius_crop.png'
image2_path = 'images\\Darius_OriginalSquare_WR.png'
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Resize image2 to match the dimensions of image1
resized_image2 = resize_image(image2, image1)

score = calculate_hog_similarity(image1, resized_image2)

print("Similarity Score:", score)"""

