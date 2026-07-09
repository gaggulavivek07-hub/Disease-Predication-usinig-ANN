import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import warnings
import joblib
joblib.dump(le, "label_encoder.pkl")

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "Testing.csv")

# Load data
data = pd.read_csv(csv_path)

# Remove redundant column if exists
if "Unnamed: 133" in data.columns:
    data.drop("Unnamed: 133", axis=1, inplace=True)

# Prepare data
X_train = data.drop("prognosis", axis=1).values
Y_train = data["prognosis"].values

le = LabelEncoder()
Y_train_encoded = le.fit_transform(Y_train)
Y_train_cat = tf.keras.utils.to_categorical(Y_train_encoded)

num_classes = Y_train_cat.shape[1]

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

# Train
history = model.fit(X_train, Y_train_cat, epochs=50)

# Accuracy plot
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

# LOSS plot
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# ---- TEST DATA ----
test_data = pd.read_csv(csv_path)
X_test = test_data.drop("prognosis", axis=1).values
Y_test = test_data["prognosis"].values
Y_test_encoded = le.transform(Y_test)
Y_test_cat = tf.keras.utils.to_categorical(Y_test_encoded)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, Y_test_cat, verbose=2)
print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)

# ---- NOW do prediction ----
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(Y_test_cat, axis=1)

# Regression Plot
plt.figure(figsize=(7,7))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([0, num_classes], [0, num_classes], 'r--')
plt.title("Regression Plot: Predicted vs Actual")
plt.xlabel("Actual Class Index")
plt.ylabel("Predicted Class Index")
plt.show()

# Error Distribution
errors = y_pred - y_true
plt.figure(figsize=(7,5))
plt.hist(errors, bins=20)
plt.title("Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.show()

# CONFUSION MATRIX
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(xticks_rotation=90, cmap="viridis")
plt.title("Confusion Matrix")
plt.show()

os.makedirs("/content/figures", exist_ok=True)  # change path if needed

# 1) Disease / prognosis distribution
prognosis_counts = data['prognosis'].value_counts().sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=prognosis_counts.values, y=prognosis_counts.index, palette="Blues_r")
plt.title("Disease (Prognosis) Distribution")
plt.xlabel("Number of Samples")
plt.ylabel("Disease")
plt.tight_layout()
plt.savefig("/content/figures/disease_distribution.png", dpi=200)
plt.show()


symptom_cols = [c for c in data.columns if c != "prognosis"]
symptom_freq = data[symptom_cols].sum().sort_values(ascending=False)


topk = 20
plt.figure(figsize=(12,6))
sns.barplot(x=symptom_freq.values[:topk], y=symptom_freq.index[:topk], palette="viridis")
plt.title(f"Top {topk} Most Frequent Symptoms")
plt.xlabel("Frequency (sum of 1's)")
plt.ylabel("Symptom")
plt.tight_layout()
plt.savefig("/content/figures/top_symptoms.png", dpi=200)
plt.show()

top_k = 25
top_symptoms = symptom_freq.index[:top_k].tolist()
coocc = data[top_symptoms].corr()
plt.figure(figsize=(10,8))
sns.heatmap(coocc, cmap="coolwarm", vmin=-1, vmax=1, annot=False)
plt.title(f"Co-occurrence / Correlation among Top {top_k} Symptoms")
plt.tight_layout()
plt.savefig("/content/figures/symptom_cooccurrence.png", dpi=200)
plt.show()

model.save("disease_model.h5")
