from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Testing.csv"
MODEL_PATH = BASE_DIR / "disease_model.keras"
ENCODER_PATH = BASE_DIR / "label_encoder.joblib"

st.set_page_config(page_title="Disease Prediction App", page_icon="🩺By Gaggula Vivek", layout="wide")

DISEASE_GUIDANCE = {
    "Fungal infection": {
        "care": [
            "Keep the affected skin dry and clean.",
            "Avoid scratching and sharing towels or clothes.",
            "Wear loose, breathable clothing.",
        ],
        "medicine": ["Antifungal cream such as clotrimazole or miconazole may help, but use as directed by a pharmacist or doctor."],
        "doctor": "Dermatologist",
        "when": "See a dermatologist if the rash spreads, becomes painful, or does not improve after a few days.",
    },
    "Allergy": {
        "care": [
            "Avoid the trigger if you can identify it.",
            "Drink plenty of water and rest.",
            "Use a cool compress for itching or swelling.",
        ],
        "medicine": ["An antihistamine such as cetirizine or loratadine may help for mild symptoms."],
        "doctor": "Allergist or General Physician",
        "when": "Seek medical help if breathing becomes difficult, the swelling worsens, or symptoms are severe.",
    },
    "GERD": {
        "care": [
            "Avoid spicy, acidic, or greasy foods.",
            "Eat smaller meals and avoid lying down soon after eating.",
            "Keep a food diary to find triggers.",
        ],
        "medicine": ["Antacids or acid reducers may help, but use them carefully and follow label instructions."],
        "doctor": "Gastroenterologist",
        "when": "See a gastroenterologist if symptoms happen often or are severe.",
    },
    "Bronchial Asthma": {
        "care": [
            "Avoid smoke, strong odors, dust, and cold air triggers.",
            "Use your prescribed inhaler exactly as directed.",
            "Keep your rescue inhaler with you if it has been prescribed.",
        ],
        "medicine": ["Use the prescribed bronchodilator or inhaler as advised by your doctor."],
        "doctor": "Pulmonologist",
        "when": "Get urgent care if you have chest tightness, severe shortness of breath, or wheezing that does not improve.",
    },
    "Hypertension": {
        "care": [
            "Reduce salt intake and avoid stress triggers.",
            "Exercise regularly and monitor your blood pressure.",
            "Rest and avoid overexertion.",
        ],
        "medicine": ["Blood pressure medicines should be prescribed and monitored by a doctor; do not self-medicate."],
        "doctor": "General Physician or Cardiologist",
        "when": "See a doctor if your blood pressure is frequently high or you have chest pain, dizziness, or vision changes.",
    },
    "Migraine": {
        "care": [
            "Rest in a quiet, dark room.",
            "Stay hydrated and avoid skipping meals.",
            "Limit screens and bright lights until the pain improves.",
        ],
        "medicine": ["Pain relief medicine may help for mild episodes, but follow the label or a doctor's advice."],
        "doctor": "Neurologist",
        "when": "Seek urgent care if you have sudden severe headache, weakness, or confusion.",
    },
    "Dengue": {
        "care": [
            "Rest well and drink plenty of fluids.",
            "Use paracetamol for fever if appropriate and safe for you.",
            "Avoid aspirin and other blood-thinning medicines unless prescribed.",
        ],
        "medicine": ["Paracetamol may help with fever and body aches; avoid self-medicating with aspirin."],
        "doctor": "General Physician or Infectious Disease Specialist",
        "when": "Get medical help immediately if you have severe abdominal pain, bleeding, vomiting, or extreme weakness.",
    },
    "Typhoid": {
        "care": [
            "Drink plenty of fluids and rest.",
            "Avoid food that may worsen stomach symptoms.",
            "Practice hand hygiene and safe food handling.",
        ],
        "medicine": ["Antibiotics are usually needed and should be prescribed by a doctor after evaluation."],
        "doctor": "General Physician or Infectious Disease Specialist",
        "when": "See a doctor promptly if you have high fever, persistent vomiting, or blood in the stool.",
    },
}

DEFAULT_GUIDANCE = {
    "care": [
        "Rest and drink enough fluids.",
        "Avoid stress and overexertion until symptoms improve.",
        "Keep a note of your symptoms and how long they last.",
    ],
    "medicine": ["Use only safe, over-the-counter symptom relief if appropriate; follow the label or ask a pharmacist."],
    "doctor": "General Physician",
    "when": "Visit a doctor if symptoms become severe, persist, or worsen quickly.",
}


def get_guidance(predicted_disease: str) -> dict:
    return DISEASE_GUIDANCE.get(predicted_disease, DEFAULT_GUIDANCE)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    data = pd.read_csv(DATASET_PATH)
    if "Unnamed: 133" in data.columns:
        data = data.drop(columns=["Unnamed: 133"])
    return data


def remove_saved_model_files():
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    if ENCODER_PATH.exists():
        ENCODER_PATH.unlink()


@st.cache_resource(show_spinner=False)
def load_or_train_model():
    data = load_dataset()
    feature_columns = [col for col in data.columns if col != "prognosis"]

    if MODEL_PATH.exists() and ENCODER_PATH.exists():
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            label_encoder = joblib.load(ENCODER_PATH)
            return model, label_encoder, feature_columns
        except Exception:
            remove_saved_model_files()

    with st.spinner("Training the disease prediction model. This will only happen once."):
        X = data[feature_columns].astype("float32").to_numpy()
        y = data["prognosis"].to_numpy()

        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_categorical = tf.keras.utils.to_categorical(y_encoded)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(X.shape[1],)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(96, activation="relu"),
                tf.keras.layers.Dense(y_categorical.shape[1], activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(X, y_categorical, epochs=25, batch_size=8, verbose=0)

        model.save(MODEL_PATH)
        joblib.dump(label_encoder, ENCODER_PATH)

    return model, label_encoder, feature_columns


model, label_encoder, feature_columns = load_or_train_model()

st.title("🩺 Disease Prediction App")
st.caption("A simple Streamlit app for predicting a disease from symptom selections.")

page = st.sidebar.radio("Navigate", ["Home", "Predict Disease", "About"])

if page == "Home":
    st.header("Welcome")
    st.write(
        "This app uses a trained neural network to suggest the most likely disease based on the symptoms you select."
    )

    data = load_dataset()
    st.metric("Dataset rows", data.shape[0])
    st.metric("Symptoms available", len(feature_columns))
    st.metric("Possible diseases", len(label_encoder.classes_))

elif page == "Predict Disease":
    st.header("Select the symptoms you notice")
    st.write("Choose one or more symptoms below and the app will predict the most likely disease.")

    symptom_inputs = {column: st.checkbox(column) for column in feature_columns}

    if st.button("Predict Disease"):
        selected_symptoms = [1 if symptom_inputs[col] else 0 for col in feature_columns]

        if sum(selected_symptoms) == 0:
            st.warning("Please select at least one symptom before predicting.")
        else:
            features = np.array(selected_symptoms, dtype="float32").reshape(1, -1)
            probabilities = model.predict(features, verbose=0)[0]
            predicted_index = int(np.argmax(probabilities))
            predicted_disease = label_encoder.inverse_transform([predicted_index])[0]

            st.success(f"Most likely disease: {predicted_disease}")

            guidance = get_guidance(predicted_disease)
            with st.expander("Suggested care and next steps", expanded=True):
                st.write("**General care**")
                for item in guidance["care"]:
                    st.write(f"- {item}")

                st.write("**Possible medicine**")
                for item in guidance["medicine"]:
                    st.write(f"- {item}")

                st.write(f"**Doctor to visit:** {guidance['doctor']}")
                st.write(f"**When to seek help:** {guidance['when']}")

            probability_df = pd.DataFrame(
                {
                    "Disease": label_encoder.classes_,
                    "Probability": np.round(probabilities * 100, 2),
                }
            ).sort_values(by="Probability", ascending=False)
            st.subheader("Probability breakdown")
            st.bar_chart(probability_df.set_index("Disease"))
            st.dataframe(probability_df.head(10), use_container_width=True)

elif page == "About":
    st.header("About this project")
    st.write(
        "This Streamlit project turns the disease dataset into an interactive predictor."
        "It demonstrates how a neural network can be trained, saved, and reused for quick predictions."
    )
    st.write("- Python")
    st.write("- Streamlit")
    st.write("- TensorFlow")
    st.write("- scikit-learn")
