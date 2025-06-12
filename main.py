from data_loader import load_data
from feature_extractor import extract_features
from train import train_svm
from evaluate import evaluate_model
from sklearn.model_selection import train_test_split
from utils import prepare_dirs

def main():
    prepare_dirs()

    print("[INFO] Loading data from Hugging Face...")
    images, labels = load_data(max_per_class=2000)

    print("[INFO] Extracting features using HOG...")
    features = extract_features(images)

    print("[INFO] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

    print("[INFO] Training SVM...")
    train_svm(X_train, y_train, model_path="models/svm_model.pkl")

    print("[INFO] Evaluating model...")
    evaluate_model(X_test, y_test, model_path="models/svm_model.pkl")

if __name__ == "__main__":
    main()
