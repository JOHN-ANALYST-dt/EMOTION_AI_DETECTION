import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Imports for the custom preprocessing function from emotions.ipynb
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Configuration ---
# Set the title and layout of the Streamlit app
st.set_page_config(
    page_title="CareEmotion AI",
    layout="centered"
)

# --- 1. Load Model and Artifacts ---

@st.cache_resource
def load_artifacts():
    """Loads the trained model, vectorizer, and emotion labels."""
    try:
        vectorizer = joblib.load("tokenizer.pkl")
        model = joblib.load('model_lr.pkl') #logistic regression model  
        emotion_labels = joblib.load('emotion_labels.pkl')
        # Initialize NLTK components for the preprocess function
        # Ensure 'stopwords' and 'wordnet' are downloaded if running in a fresh environment
        # nltk.download(['stopwords', 'wordnet']) 
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        return vectorizer, model, emotion_labels, stop_words, lemmatizer
    except FileNotFoundError as e:
        st.error(f"Error: Model file not found. Please ensure 'tfidf_vectorizer.pkl', 'logistic_regression_model.pkl', and 'emotion_labels.pkl' are in the same directory.")
        st.stop()

vectorizer, model, emotion_labels, stop_words, lemmatizer = load_artifacts()

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002500-\U00002BEF"  # chinese chars
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)
# --- 2. Custom Preprocessing Function (from emotions.ipynb) ---
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
    # Using simple split() since that's what was in the original notebook code
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    
    return " ".join(tokens)

# --- 3. Prediction and Display Logic ---

def predict_emotion(text):
    """Predicts emotions for the given text."""
    
    # 1. Preprocess the input text
    clean_text = preprocess(text)
    
    # 2. Vectorize the clean text
    # The vectorizer expects a list of documents
    text_vectorized = vectorizer.transform([clean_text])
    
    # 3. Get raw prediction (0 or 1 for each label)
    prediction_raw = model.predict(text_vectorized)[0]
    
    # 4. Get probability estimates (from the underlying classifier)
    # This is a good way to show 'strength' of emotion
    all_probas = []
    for i, estimator in enumerate(model.estimators_):
        # Predict_proba returns [prob_class_0, prob_class_1]
        proba = estimator.predict_proba(text_vectorized)[:, 1] 
        all_probas.append(proba[0])
    
    # Create a DataFrame for display
    results_df = pd.DataFrame({
        'Emotion': emotion_labels,
        'Predicted': prediction_raw,
        'Confidence (%)': np.round(np.array(all_probas) * 100, 2)
    })
    
    # Filter only for predicted (1) or high-confidence (e.g., > 10%) emotions
    # High-confidence filter helps if the model is too conservative
    results_df = results_df[
        (results_df['Predicted'] == 1) | (results_df['Confidence (%)'] > 10)
    ]
    
    # Sort by confidence
    results_df = results_df.sort_values(by='Confidence (%)', ascending=False)
    
    return results_df


# --- 4. Streamlit UI Design ---

st.title("🧠 CareEmotion  Multi-Label Detector")
st.markdown("Enter a piece of text (like a Reddit comment) to classify the emotions it contains.")

# Text input widget
user_input = st.text_area(
    "Enter Text Here:",
    "I was so excited when I saw the final result, but then I felt a little sad for the losers."
)

if st.button("Analyze Emotion"):
    if user_input:
        # Get the prediction results
        prediction_results = predict_emotion(user_input)
        
        # --- Display Results ---
        
        st.subheader("Analysis Complete")

        # Display the primary predicted emotions
        predicted_emotions = prediction_results[prediction_results['Predicted'] == 1]['Emotion'].tolist()

        if predicted_emotions:
            st.success(f"**Primary Emotions Detected:** {', '.join([e.title() for e in predicted_emotions])}")
        else:
            # If the model didn't predict a '1' for any label (rare for a multi-label classifier)
            st.info("No strong single emotion predicted. Showing top confidence scores.")

        # Display the full confidence breakdown
        st.markdown("#### Confidence Breakdown")
        st.dataframe(
            prediction_results.drop(columns=['Predicted']), # Don't show the raw 0/1 prediction
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