import os

def prepare_dirs():
    if not os.path.exists("models"):
        os.makedirs("models")
