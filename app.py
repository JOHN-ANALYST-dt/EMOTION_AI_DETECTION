import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import string
import nltk

# --- NLTK Data Download Fix ---
# CRITICAL: Fixes the LookupError by correctly downloading resources 
# for a fresh deployment environment.
try:
    # Attempt to load the required data first
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError: 
    # If the resource is not found, download them.
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Imports must happen AFTER the downloads, otherwise they will fail
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# --- End NLTK Fix ---

# --- Configuration ---


PREDICTED_EMOTIONS = [
    'anger', 
    'disgust', 
    'fear', 
    'joy', 
    'sadness', 
    'surprise'
]
# --- CSS INJECTION FUNCTION ---
def inject_custom_css(file_path):
    """Reads a local CSS file and injects it into the Streamlit app."""
    try:
        with open(file_path) as f:
            # st.markdown injects the CSS wrapped in <style> tags
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found at path: {file_path}")
    except Exception as e:
        st.error(f"Error injecting CSS: {e}")

# Call the function with the correct path to apply styles immediately
inject_custom_css("style.css")
st.set_page_config(
    page_title="CareEmotion AI",
    layout="centered"
)

# --- 1. Load Model and Artifacts ---

@st.cache_resource
def load_artifacts():
    """Loads the trained model, vectorizer, and emotion labels."""
    try:
        # NOTE: Using the file names as they appear in your code
        vectorizer = joblib.load("tokenizer.pkl") 
        model = joblib.load('model_lr.pkl') 
        emotion_labels = PREDICTED_EMOTIONS
        
        # Initialize NLTK components now that the resources are guaranteed to be downloaded
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        return vectorizer, model, emotion_labels, stop_words, lemmatizer
        
    except FileNotFoundError:
        st.error("""
            **Deployment Error: Model files not found!**
            Please ensure these files are in your repository root and committed:
            - `tokenizer.pkl`
            - `model_lr.pkl` 
            - `emotion_labels.pkl`
        """)
        st.stop()

# Load the resources globally
vectorizer, model, emotion_labels, stop_words, lemmatizer = load_artifacts()

# --- Pre-compile Regex (outside of function for efficiency) ---
emoji_pattern = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002500-\U00002BEF\U00002702-\U000027B0\U000024C2-\U0001F251]+",
    flags=re.UNICODE
)

# --- 2. Custom Preprocessing Function ---
def preprocess(text):
    """
    Cleans and preprocesses text using the same logic as the training notebook.
    """
    # remove emoji
    text = emoji_pattern.sub(r'', text)

    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove leading/trailing quotes often found in notebook snippets
    text = text.strip('\"')
    
    # Lowercase and remove punctuation
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize, lemmatize, and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    
    return " ".join(tokens)

# --- 3. Prediction and Display Logic ---

def predict_emotion(text):
    """Predicts emotions for the given text."""
    
    # 1. Preprocess the input text
    clean_text = preprocess(text)
    
    # 2. Vectorize the clean text
    text_vectorized = vectorizer.transform([clean_text])
    
    # 3. Get raw prediction (0 or 1 for each label)
    prediction_raw = model.predict(text_vectorized)[0]
    
    # 4. Get probability estimates (from the underlying classifier)
    all_probas = []
    if hasattr(model, 'estimators_'):
        for i, estimator in enumerate(model.estimators_):
            proba = estimator.predict_proba(text_vectorized)[:, 1] 
            all_probas.append(proba[0])
    else:
        st.error("Model structure not recognized. Cannot extract confidence scores.")
        return None
    # Add validation check for array lengths
    len_labels = len(emotion_labels)
    len_probas = len(all_probas)
    len_predictions = len(prediction_raw)
    
    if not (len_labels == len_probas == len_predictions):
        st.error(f"""
            **Data Length Mismatch Error: Cannot Create Results Table**
            The number of emotion labels does not match the model's output size.
            - Length of `emotion_labels.pkl`: {len_labels}
            - Length of Model Predictions: {len_predictions}
            
            **Action Required:** You must re-save your `emotion_labels.pkl` file to ensure it contains the exact same number of labels ({len_predictions}) that your `model_lr.pkl` was trained to predict.
        """)
        return None
    # Create a DataFrame for display
    results_df = pd.DataFrame({
        'Emotion': emotion_labels,
        'Predicted': prediction_raw,
        'Confidence (%)': np.round(np.array(all_probas) * 100, 2)
    })
    
    # Filter only for predicted (1) or high-confidence (e.g., > 10%) emotions
    results_df = results_df[
        (results_df['Predicted'] == 1) | (results_df['Confidence (%)'] > 10)
    ]
    
    results_df = results_df.sort_values(by='Confidence (%)', ascending=False)
    
    return results_df


# --- 4. Streamlit UI Design ---

st.title("🧠 CareEmotion Multi-Label Detector")
st.markdown("Enter a piece of text (like a Reddit comment) to classify the emotions it contains.")

# Text input widget
user_input = st.text_area(
    "Enter Text Here:",
    placeholder="e.g., I am happy on 7/10 days",
    height=150
)

# 2. Add an Analyze Button to trigger the check
if st.button("Check Input"):
    
    # 3. Check for Empty Input
    if not user_input.strip():
        st.warning("Input is empty. Please enter some text.")
    
    # 4. Check if the input contains ONLY numeric characters
    # We use .isnumeric() on the stripped input for the check.

    elif user_input.strip().isnumeric():
        # Display the warning message to the user on the web page (using st.error or st.warning)
        st.error("Input must contain text, not only numbers. Please try again.")
    
    # 5. The input is valid (contains text, potentially mixed with numbers)
    else:
        st.success("Thank you, your input is valid!")
        
        # Optionally display the validated input
        st.write("Validated Input:")
        st.code(user_input)


if st.button("Analyze Emotion"):
    if user_input:
        prediction_results = predict_emotion(user_input)
        
        if prediction_results is not None:
            # --- Display Results ---
            st.subheader("Analysis Complete")
            
            # Display the primary predicted emotions
            predicted_emotions = prediction_results[prediction_results['Predicted'] == 1]['Emotion'].tolist()

            if predicted_emotions:
                st.success(f"**Primary Emotions Detected:** {', '.join([e.title() for e in predicted_emotions])}")
            else:
                st.info("No strong single emotion predicted. Showing top confidence scores.")

            # Display the full confidence breakdown
            st.markdown("#### Confidence Breakdown")
            st.dataframe(
                prediction_results.drop(columns=['Predicted']),
                hide_index=True
            )
            
            # Optional: Display a bar chart of the top 10 confidences
            top_confidences = prediction_results.head(10)
            st.bar_chart(
                top_confidences,
                x='Emotion',
                y='Confidence (%)',
                color="#E6652B" 
            )

    else:
        st.warning("Please enter some text to analyze.")

# Add a footer/info section
st.markdown("---")
st.markdown(
    "This model is a multi-label classifier trained on the GoEmotions dataset."
)