import joblib
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(X_test, y_test, model_path='svm_model.pkl'):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
