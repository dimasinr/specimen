import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(gray_image)
    normalized_image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    print(normalized_image)
    return normalized_image

def resize_image(image, target_size):
    resize = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resize

def compare_images(image1, image2):
    difference = cv2.absdiff(image1, image2)
    similarity_score = np.sum(difference) / (image1.shape[0] * image1.shape[1])
    return similarity_score

def similarity_rating(similarity_score):
    print(int(similarity_score))
    rating = max(100 - similarity_score, 0)
    # rating = max(100 - similarity_score * 100, 0)
    # rating = 100 - similarity_score * 100
    return int(rating)

signature_image1_path = 'signature1.png'
signature_image2_path = 'signature2.png'

signature1 = preprocess_image(signature_image1_path)
signature2 = preprocess_image(signature_image2_path)

target_size = (min(signature1.shape[1], signature2.shape[1]), min(signature1.shape[0], signature2.shape[0]))
signature1 = resize_image(signature1, target_size)
signature2 = resize_image(signature2, target_size)

similarity_score = compare_images(signature1, signature2)

rating = similarity_rating(similarity_score)
print("Penilaian Kemiripan antara dua gambar tanda tangan:", rating, "%")
