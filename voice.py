import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import string
import nltk
import base64
import os
import io
import wave
# --- NEW IMPORTS for Voice Recognition ---
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from io import StringIO
import sys
# --- End NEW IMPORTS ---


# --- NLTK Data Download Fix ---
# CRITICAL: Fixes the LookupError by correctly downloading resources 
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
MODEL_EXISTS = True # Assume true until load_artifacts proves otherwise


# --- BASE64 IMAGE FUNCTIONS (Placeholder images as local files are not provided) ---
# NOTE: Since actual image files ('images/happy2.jpg', 'images/adventist_logo.png') 
# were not available, the image loading functions are temporarily disabled 
# to prevent FileNotFoundError, and placeholders are used. 
# You should restore this if you provide the image files.
def get_base64_image(image_path, mime_type="image/jpeg"):
    """Converts a local image file to a Base64 string."""
    try:
        if not os.path.exists(image_path):
             # Placeholder logic: return a simple base64 placeholder or None
            return None # Returning None simplifies the UI rendering checks below
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded}" 
    except Exception as e:
        return None

# Load your local images
# bg_image = get_base64_image("images/happy2.jpg") # Assuming you want to use this for the title background
# ADVENTIST_LOGO_URI = get_base64_image("images/adventist_logo.png", mime_type="image/png")
# Simplified URI loading to prevent errors if files are missing
bg_image = None
ADVENTIST_LOGO_URI = None

# --- CSS INJECTION FUNCTION ---
def inject_custom_css(file_path):
    """Reads a local CSS file and injects it into the Streamlit app."""
    try:
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # st.error(f"CSS file not found at path: {file_path}") # Removed error for clean output
        pass 
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
    global MODEL_EXISTS
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
        MODEL_EXISTS = False
        st.error("""
            **Deployment Error: Model files not found!**
            Please ensure these files are in your repository root and committed:
            - `tokenizer.pkl`
            - `model_lr.pkl` 
        """)
        # Still return NLTK components for preprocessing function to work, even if prediction fails
        return None, None, PREDICTED_EMOTIONS, set(stopwords.words('english')), WordNetLemmatizer()

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
    Uses globally loaded NLTK components (stop_words, lemmatizer).
    """
    if pd.isna(text) or text is None: return ""
    text = str(text)

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
    
    if not MODEL_EXISTS: return None
    
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

    len_labels = len(emotion_labels)
    len_probas = len(all_probas)
    len_predictions = len(prediction_raw)
    
    if not (len_labels == len_probas == len_predictions):
        st.error("Data Length Mismatch Error: Cannot Create Results Table.")
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
    ].sort_values(by='Confidence (%)', ascending=False)
    
    return results_df


# =======================================================
# --- VOICE PROCESSING FUNCTIONS (NEW) ---
# =======================================================

def raw_pcm_to_wav_bytes(raw_audio_data, sample_rate=44100, num_channels=1):
    """Converts a raw NumPy array (s16le) to a WAV file in BytesIO."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # 16-bit audio (2 bytes per sample)
        wf.setframerate(sample_rate)
        # raw_audio_data is 1D for mono, need to ensure it's in the right format
        if raw_audio_data.dtype != np.int16:
            raw_audio_data = raw_audio_data.astype(np.int16)
        wf.writeframes(raw_audio_data.tobytes())
    return buffer.getvalue()

def transcribe_audio(wav_bytes):
    """Transcribes audio data using Google's free Web Speech API."""
    r = sr.Recognizer()
    
    # Use AudioFile class which can read from a file-like object (BytesIO)
    with io.BytesIO(wav_bytes) as audio_io:
        try:
            with sr.AudioFile(audio_io) as source:
                # Adjust for ambient noise and capture the audio data
                audio = r.record(source)  

            # Use the Google Web Speech API for transcription
            text = r.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio. Please speak clearly."
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"
        except Exception as e:
            return f"An unexpected error occurred during audio processing: {e}"

# --- Audio Processor Class for Streamlit-webrtc ---
class AudioAnalysisProcessor(AudioProcessorBase):
    def __init__(self):
        # We store the raw PCM frames
        self.audio_frames = []
        
    def recv(self, frame):
        # frame.to_ndarray() converts the frame to a NumPy array (s16le, 16-bit PCM)
        self.audio_frames.append(frame.to_ndarray(format="s16le")) 
        return frame

# =======================================================
# --- Sidebar UI (Voice Recognition & Preprocessor Navigation) ---
# =======================================================

# --- Voice Recording and Analysis (NEW SECTION, Aligned Left) ---
st.sidebar.subheader("🎙️ Voice Analyzer (STT)")
st.sidebar.markdown(
    """
    <p style='font-size: small; color: #888;'>
    Use your microphone to record speech for text analysis.
    </p>
    """, unsafe_allow_html=True
)

voice_results_placeholder = st.sidebar.empty()

# --- UI for Voice Recording ---
ctx = webrtc_streamer(
    key="speech_emotion_detector",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioAnalysisProcessor,
    media_stream_constraints={"video": False, "audio": True},
    async_processing=True,
    # Place in sidebar by defining the container context
    container=st.sidebar
)

# Logic to run analysis after recording stops
if st.sidebar.button("Analyze Recorded Voice", use_container_width=True, key="voice_analyze_btn"):
    
    # Clear previous results
    voice_results_placeholder.empty()

    if not ctx or not ctx.audio_processor or not ctx.audio_processor.audio_frames:
        voice_results_placeholder.warning("Please record some voice audio first.")
        
    else:
        with voice_results_placeholder.container():
            with st.spinner("Processing audio..."):
                
                # 1. Concatenate the audio chunks into one numpy array
                raw_audio_data = np.concatenate(ctx.audio_processor.audio_frames, axis=0)
                
                # 2. Convert PCM data to WAV format bytes
                wav_bytes = raw_pcm_to_wav_bytes(raw_audio_data)
                
                # 3. Transcribe the audio
                transcribed_text = transcribe_audio(wav_bytes)
                
                st.subheader("Transcription")
                
                if "Could not understand audio" in transcribed_text or "Could not request results" in transcribed_text:
                    st.error(transcribed_text)
                
                else:
                    st.success("Transcription successful!")
                    st.caption("Text:")
                    st.code(transcribed_text, language='text')
                    
                    # 4. Analyze the transcribed text using existing tools
                    st.markdown("---")
                    st.subheader("Emotion Analysis")
                    
                    prediction_results = predict_emotion(transcribed_text)
                    
                    if prediction_results is not None and not prediction_results.empty:
                        predicted_emotions = prediction_results[prediction_results['Predicted'] == 1]['Emotion'].tolist()
                        
                        if predicted_emotions:
                            st.success(f"**Detected:** {', '.join([e.title() for e in predicted_emotions])}")
                        else:
                            st.info("No strong single emotion predicted.")

                        st.markdown("##### Confidence")
                        st.dataframe(
                            prediction_results[['Emotion', 'Confidence (%)']].head(3),
                            hide_index=True
                        )
                        
                    else:
                        st.warning("Text classification model not available or text was too short/unclear.")
        
# --- Separator for Preprocessor Tools ---
st.sidebar.markdown("---")
st.sidebar.header("🛠️ Preprocessing Tools")
 
# --- MAIN CONTENT UI ---

# --- DYNAMIC TITLE SECTION (using Base64 image) ---
if bg_image and ADVENTIST_LOGO_URI:
    # Existing dynamic title HTML (assuming it was correctly using the variables)
    st.markdown(
        f"""
        <style>
        /* ... existing CSS styles for main-title-container, etc. ... */
        </style>

        <div class="main-title-container">
            <img class="main-title-logo" src="{ADVENTIST_LOGO_URI}" alt="Adventist Logo"> 
            <h1 class="main-title-text">CareEmotion AI</h1>
            <p class="main-title-caption">Advanced Multi-label Sentiment and Emotion Analysis</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
else:
    # Fallback title (which will be used if images are missing)
    st.markdown(
        """
        <div class="main-title">
            <div><h4>Pride lands SDA</h4></div>
            <h1>CareEmotion AI</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class="caption-intro">
        Understand Emotions in Text
        <h4>
        Our AI model analyzes text to detect emotional undertones with precision
        </h4>
    </div>
    """, 
    unsafe_allow_html=True
)
st.markdown(
    '<div class="arrow-pointer">⬇️ Type your text below ⬇️</div>',
    unsafe_allow_html=True
)

# --- Text Input Emotion Analysis ---
user_input = st.text_area(
    "Enter Text Here:",
    placeholder="e.g., Hello church! God is good😊",
    height=150,
    key="main_text_input" # Added key for uniqueness
)

st.markdown(
    """
    <script>
    // Apply custom CSS class to Streamlit textarea
    var textareas = window.parent.document.querySelectorAll('textarea');
    if (textareas.length > 0) {
        // Only target the last one if you have multiple textareas, 
        // but adding a key is better practice in Streamlit
        // textareas[textareas.length - 1].classList.add('custom-textarea'); 
    }
    </script>
    """,
    unsafe_allow_html=True
)

# 2. Add an Analyze Button to trigger the check
if st.button("Check Input"):
    
    if not user_input.strip():
        st.warning("Input is empty. Please enter some text.")
    
    elif user_input.strip().isnumeric():
        st.error("Input must contain text, not only numbers. Please try again.")
    
    else:
        st.success("Thank you, your input is valid!")
        st.write("Validated Input:")
        st.code(user_input)


if st.button("Analyze Emotion", type="primary"):
    if user_input:
        if not MODEL_EXISTS:
            st.error("Cannot analyze emotion: Model files are missing.")
        else:
            prediction_results = predict_emotion(user_input)
            
            if prediction_results is not None:
                # --- Display Results ---
                st.subheader("Text Analysis Complete")
                
                predicted_emotions = prediction_results[prediction_results['Predicted'] == 1]['Emotion'].tolist()

                if predicted_emotions:
                    st.success(f"**Primary Emotions Detected:** {', '.join([e.title() for e in predicted_emotions])}")
                else:
                    st.info("No strong single emotion predicted. Showing top confidence scores.")

                st.markdown("#### Confidence Breakdown")
                st.dataframe(
                    prediction_results.drop(columns=['Predicted']),
                    hide_index=True
                )
                
                top_confidences = prediction_results.head(10)
                st.bar_chart(
                    top_confidences,
                    x='Emotion',
                    y='Confidence (%)',
                    color="#E6652B" 
                )

    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")

# --- 5. Data Preprocessing UI Functions ---

def ui_handle_missing_values(df_key):
    """
    UI and logic for handling missing values in the DataFrame.
    """
    df = st.session_state[df_key]
    st.subheader("🧹 1. Handle Missing Values")
    
    # Show initial state
    missing_summary = df.isnull().sum()
    missing_data = missing_summary[missing_summary > 0]
    
    if missing_data.empty:
        st.success("✅ No missing values found in the DataFrame!")
    else:
        st.warning(f"⚠️ Found missing values in {len(missing_data)} column(s).")
        st.dataframe(missing_data.rename("Missing Count"))
        
        st.markdown("#### Select Strategy")
        
        col_to_clean = st.selectbox(
            "Select Column to Handle Missing Values (Only columns with NaNs shown):",
            options=["-- Select a Column --"] + missing_data.index.tolist(),
            key='missing_col_select'
        )
        
        if col_to_clean != "-- Select a Column --":
            strategy = st.radio(
                f"Choose a strategy for **{col_to_clean}**:",
                options=["Drop Rows", "Fill with a value (e.g., 'unknown')", "Fill with Mode/Mean/Median (for numeric/categorical)"],
                key='missing_strategy_radio'
            )
            
            if st.button("Apply Missing Value Strategy", key='apply_missing_btn'):
                new_df = df.copy() # Work on a copy
                
                if strategy == "Drop Rows":
                    new_df.dropna(subset=[col_to_clean], inplace=True)
                    st.session_state[df_key] = new_df
                    st.success(f"Successfully dropped rows with missing values in **{col_to_clean}**.")
                    st.dataframe(new_df.head())
                    
                elif strategy == "Fill with a value (e.g., 'unknown')":
                    fill_value = st.text_input("Enter value to fill NaNs with:", value="unknown", key='fill_value_input')
                    new_df[col_to_clean].fillna(fill_value, inplace=True)
                    st.session_state[df_key] = new_df
                    st.success(f"Successfully filled missing values in **{col_to_clean}** with **'{fill_value}'**.")
                    st.dataframe(new_df.head())
                
                # Add more complex strategies here if needed

def ui_clean_text_data(df_key):
    """
    UI and logic for applying the defined text preprocessing function 
    to a selected column in the DataFrame.
    """
    df = st.session_state[df_key]
    st.subheader(" 2. Text Cleaning & Normalization")
    st.markdown("Applies the full preprocessing pipeline: remove emojis, HTML, numbers, lowercase, remove punctuation, lemmatization, and stop word removal.")
    
    # Identify text columns (simplistic check)
    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if not text_columns:
        st.error("No suitable text columns (object dtype) found in the DataFrame.")
        return
    
    col_to_clean = st.selectbox(
        "Select Text Column to Clean:",
        options=text_columns,
        key='clean_col_select'
    )
    
    # Option to create a new column or overwrite
    cleaning_action = st.radio(
        "Cleaning Action:",
        options=["Create New Column", "Overwrite Existing Column"],
        key='cleaning_action_radio'
    )
    
    new_col_name = col_to_clean + "_clean"
    if cleaning_action == "Create New Column":
        new_col_name = st.text_input("New Column Name:", value=new_col_name, key='new_col_name_input')

    
    if st.button(f"Apply Preprocessing to **{col_to_clean}**", key='apply_clean_btn'):
        try:
            with st.spinner(f"Cleaning text in column **{col_to_clean}**... This may take a moment."):
                
                # Apply the preprocessing function (uses global artifacts)
                processed_series = df[col_to_clean].apply(preprocess)

            # Update the DataFrame in session state
            if cleaning_action == "Overwrite Existing Column":
                st.session_state[df_key][col_to_clean] = processed_series
                st.success(f"Successfully **overwrote** column **{col_to_clean}** with cleaned text.")
            else:
                st.session_state[df_key][new_col_name] = processed_series
                st.success(f"Successfully created new column **{new_col_name}** with cleaned text.")

            # Show a preview of the changes
            st.markdown("#### Preview of Cleaned Data")
            if cleaning_action == "Overwrite Existing Column":
                st.dataframe(st.session_state[df_key][[col_to_clean]].head(10))
            else:
                # Ensure all columns exist before trying to display them
                cols_to_display = [col_to_clean, new_col_name] if new_col_name in st.session_state[df_key].columns else [col_to_clean]
                st.dataframe(st.session_state[df_key][cols_to_display].head(10))

        except Exception as e:
            st.error(f"An error occurred during cleaning: {e}")


# --- Data Preprocessor App ---
with st.container(border=True):
    st.header(" Dedicated Data Preprocessor Tool")

st.title(" CSV Text Data Preprocessor")
st.markdown(
    '<div class="caption-intro">Upload your CSV file and use the sidebar menu to apply preprocessing steps.</div>',
    unsafe_allow_html=True
)

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a CSV file...", type="csv")

if uploaded_file is not None:
    # 1. Initialization and Data Storage
    df_key = 'current_df'
    
    # Check if a new file was uploaded or if the session is fresh
    if df_key not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded_file.name:
        try:
            st.session_state[df_key] = pd.read_csv(uploaded_file)
            st.session_state['uploaded_file_name'] = uploaded_file.name
            st.success(f"CSV file '{uploaded_file.name}' loaded successfully! Use the sidebar to begin.")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            del st.session_state[df_key]
            st.stop()
            
    df = st.session_state[df_key]
    
    st.write("---")
    
    # --- Sidebar Navigation (The Dropdown Menu) ---
    # The header for tools is already in the sidebar above the voice section.
    
    # Use a dropdown to select the current task
    processing_step = st.sidebar.selectbox(
        "Select a step:",
        options=[
            "View Raw Data",
            "1. Handle Missing Values", 
            "2. Text Cleaning & Normalization"
        ],
        key='processing_step_select'
    )
    
    st.sidebar.markdown("---")
    
    # --- Main Content Area Logic ---
    
    if processing_step == "View Raw Data":
        st.subheader("🔍 Current Data View")
        st.info(f"Loaded DataFrame has {df.shape[0]} rows and {df.shape[1]} columns.")
        st.dataframe(df.head(20))
        st.write("Use the sidebar menu to select a preprocessing task.")
        
    elif processing_step == "1. Handle Missing Values":
        ui_handle_missing_values(df_key)
        
    elif processing_step == "2. Text Cleaning & Normalization":
        ui_clean_text_data(df_key)


    # --- Download Section (Always Visible) ---
    st.markdown("---")
    st.subheader("⬇️ Download Result")
    st.write("Download the CSV with all applied modifications.")
    
    csv_buffer = StringIO()
    # Use the modified DataFrame from session state
    st.session_state[df_key].to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="Download Cleaned CSV",
        data=csv_buffer.getvalue(),
        file_name=f"cleaned_{st.session_state['uploaded_file_name']}",
        mime="text/csv",
    )
    
else:
    st.info("Awaiting CSV file upload.")

# Add a footer/info section
st.markdown("---")
st.markdown(
    "This model is a multi-label classifier trained on the GoEmotions dataset."
)