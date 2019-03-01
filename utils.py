import numpy as np
from flask import g, request
from src.VAE import VAE
from src.align import SSDAligner

def allowed_file(uploaded_file, allowed_extensions):
    filename = uploaded_file.filename
    if "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in allowed_extensions

def get_model():
    if "model" not in g:
        g.model = VAE("./src/checkpoints/20170708-150701/model.ckpt-50000")
    return g.model

def get_aligner():
    if "aligner" not in g:
        g.aligner = SSDAligner()
    return g.aligner

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
def get_uploaded_image(name="image"):
    if name not in request.files:
        return None
    uploaded_file = request.files[name]
    if not uploaded_file.filename:
        return None
    if not allowed_file(uploaded_file, ALLOWED_EXTENSIONS):
        print("Uploaded file is not an image")
        return None
    return uploaded_file

def read_image_file(image_file):
    byte_array = np.fromstring(image_file.read(), dtype=np.uint8)
    image = cv2.imdecode(byte_array, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
