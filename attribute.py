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
    parser.add_argument("attribute", help="which attribute vector to use")
    args = parser.parse_args()

    CHECKPOINT_PATH = "./src/checkpoints/20170708-150701/model.ckpt-50000"
    CHECKPOINT_PATH = os.path.abspath(CHECKPOINT_PATH)
    ATTRIBUTE_VECTORS_PATH = "./src/checkpoints/attribute_vectors.h5"
    ATTRIBUTE_VECTORS_PATH = os.path.abspath(ATTRIBUTE_VECTORS_PATH)

    image_path = "./images/robert-greene.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    aligner = SSDAligner()
    face = aligner.align_and_crop_face(image, image_dim=64)
    face = np.expand_dims(face, axis=0)
    model = VAE(CHECKPOINT_PATH)
    code = model.get_code(face)

    with open("./attribute_vectors.pickle", "rb") as f:
        data = pickle.load(f)
        attributes_dict = data["attributes_dict"]
        attribute_vectors = data["attribute_vectors"]

    attribute_index = attributes_dict[args.attribute]
    coefficients = np.arange(20).reshape((20, 1)) * 0.25
    attribute_vector = attribute_vectors[attribute_index].reshape((1, 100))
    new_codes = coefficients @ attribute_vector + code

    reconstructions = model.get_reconstruction_from_code(new_codes)
    for i, reconstruction in enumerate(reconstructions):
        bgr_reconstruction = cv2.cvtColor(reconstruction, cv2.COLOR_RGB2BGR)
        resized_reconstruction = cv2.resize(bgr_reconstruction, (250, 250))
        cv2.imwrite(f"./output/attribute/{i}.jpg", resized_reconstruction)
