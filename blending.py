import os
import argparse
import numpy as np
import cv2
import h5py
from src.VAE import VAE
from src.align import SSDAligner
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image1_path", help="Path to first image.",
                        default="./images/daniel-radcliffe.jpg")
    parser.add_argument("--image2_path", help="Path to second image.",
                        default="./images/jason-chen.jpg")
    args = parser.parse_args()

    CHECKPOINT_PATH = "./src/checkpoints/20170708-150701/model.ckpt-50000"
    CHECKPOINT_PATH = os.path.abspath(CHECKPOINT_PATH)
    ATTRIBUTE_VECTORS_PATH = "./src/checkpoints/attribute_vectors.h5"
    ATTRIBUTE_VECTORS_PATH = os.path.abspath(ATTRIBUTE_VECTORS_PATH)

    image = cv2.imread(args.image1_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread(args.image2_path)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    aligner = SSDAligner()
    face = aligner.align_and_crop_face(image, image_dim=64)
    face2 = aligner.align_and_crop_face(image2, image_dim=64)
    faces = np.stack([face, face2])

    model = VAE(CHECKPOINT_PATH)
    codes = model.get_code(faces)

    difference_vector = codes[1] - codes[0]

# Interpolate
    difference_vector = difference_vector.reshape((1, 100))
    coefficients = np.linspace(0, 1, 20).reshape((20, 1))
    new_codes = coefficients @ difference_vector + codes[0]

    reconstructions = model.get_reconstruction_from_code(new_codes)
    for i, reconstruction in enumerate(reconstructions):
        bgr_reconstruction = cv2.cvtColor(reconstruction, cv2.COLOR_RGB2BGR)
        resized_reconstruction = cv2.resize(bgr_reconstruction, (250, 250))
        cv2.imwrite(f"./output/blending/{i}.jpg", resized_reconstruction)
