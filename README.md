# 🐱🐶 Cat vs Dog Image Classifier using SVM

This project classifies cat and dog images using the **Support Vector Machine (SVM)** algorithm. It uses **HOG (Histogram of Oriented Gradients)** for feature extraction and the `microsoft/cats_vs_dogs` dataset from Hugging Face.

---

## 🔧 Technologies Used
- Python 3.13
- scikit-learn
- scikit-image
- Hugging Face Datasets
- OpenCV
- NumPy
- Pillow (PIL)
- Joblib

---

## 📁 Project Structure
svm_cat_dog_classifier/

├── data_loader.py # Loads dataset from Hugging Face

├── feature_extractor.py # Extracts HOG features

├── train.py # Trains the SVM classifier

├── evaluate.py # Evaluates the classifier

├── utils.py # Helper functions

├── main.py # Runs the full pipeline

├── models/ # Saved model (svm_model.pkl)

└── README.md # Project documentation

---

## 🚀 How to Run in VS Code Terminal

```bash
1. python -m venv venv
2. source venv/bin/activate  # On Mac/Linux
   venv\Scripts\activate     # On Windows

3. pip install -r requirements.txt

4. python main.py
```
---

### 📊 Final Output
Once you run main.py, the following will be printed:

✅ Validation Accuracy

📋 Classification Report (Precision, Recall, F1-Score)

🔢 Confusion Matrix

Sample Console Output

[INFO] Loading data from Hugging Face...

[INFO] Extracting features using HOG...

[INFO] Splitting data...

[INFO] Training SVM...

[INFO] Model trained. Saving to models/svm_model.pkl

[INFO] Evaluating model...

Classification Report:
               precision    recall  f1-score   support

        Cat        0.87      0.84      0.85       250
        Dog        0.85      0.88      0.86       250

    accuracy                         0.86       500
Confusion Matrix:
[[210  40]
 [ 30 220]]
