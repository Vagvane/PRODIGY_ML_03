from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_svm(X_train, y_train, model_path='svm_model.pkl'):
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    print(f"[INFO] Model trained. Saving to {model_path}")
    joblib.dump(model, model_path)
