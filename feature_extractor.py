from skimage.feature import hog
import numpy as np

def extract_features(images):
    features = []
    for img in images:
        feature = hog(img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys')
        features.append(feature)
    return np.array(features)
