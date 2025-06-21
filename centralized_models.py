
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras import models, layers, utils
import json

# Load dataset
df = pd.read_csv("synthetic_tcga_brca.csv")
X = df.drop('Subtype', axis=1).values
y = df['Subtype'].values

# Normalize gene expression values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality reduction using PCA
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_scaled)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, stratify=y, random_state=42)

# Initialize models
models_dict = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}

# Train and evaluate traditional ML models
for name, model in models_dict.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(utils.to_categorical(y_test), y_prob, multi_class='ovr')
    results[name] = {
        "classification_report": report,
        "auc": auc
    }

# CNN model input preparation (reshape to 10x10x1 for simplicity)
X_cnn = X_pca.reshape(-1, 10, 10, 1)
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X_cnn, y, test_size=0.2, stratify=y, random_state=42)

y_train_cat = utils.to_categorical(y_train_cnn, num_classes=4)
y_test_cat = utils.to_categorical(y_test_cnn, num_classes=4)

# Define CNN model
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(10, 10, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train CNN
cnn_model.fit(X_train_cnn, y_train_cat, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate CNN
y_pred_cnn = cnn_model.predict(X_test_cnn)
y_pred_labels = np.argmax(y_pred_cnn, axis=1)
report_cnn = classification_report(y_test_cnn, y_pred_labels, output_dict=True)
auc_cnn = roc_auc_score(y_test_cat, y_pred_cnn, multi_class='ovr')

results["CNN"] = {
    "classification_report": report_cnn,
    "auc": auc_cnn
}

# Save results to file
with open("classification_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Training complete. Results saved to classification_results.json.")
