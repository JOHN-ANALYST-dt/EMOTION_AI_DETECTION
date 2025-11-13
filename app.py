import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import string
import nltk
import os
import io
from io import StringIO
import sys
import wave

#voice processing libraries
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from io import StringIO
import sys

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

# --- intervention------
# Assuming this file exists and contains the display_interventions function
try:
    from intervention import display_interventions
except ImportError:
    # Define a placeholder function if the file is missing to prevent errors
    def display_interventions(results):
        st.info("Intervention file missing. Cannot display suggestions.")


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
        # NOTE: Since `style.css` is not provided, this line is kept as is
        # but the file must be present in the execution environment.
        with open(file_path) as f:
            # st.markdown injects the CSS wrapped in <style> tags
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # Instead of erroring out, just skip CSS if not found in this context
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
        # Placeholder error for deployment environment
        st.warning("Model files not found (tokenizer.pkl or model_lr.pkl). Analysis functions are disabled.")
        # Return mock artifacts to allow the rest of the app to run
        class MockModel:
             def predict(self, X): return np.array([[0]*len(PREDICTED_EMOTIONS)])
             def estimators_(self): return [type('MockEstimator', (), {'predict_proba': lambda self, X: np.array([[0.5, 0.5]])})() for _ in PREDICTED_EMOTIONS]
        
        class MockVectorizer:
             def transform(self, X): return None
             
        return MockVectorizer(), MockModel(), PREDICTED_EMOTIONS, set(stopwords.words('english')), WordNetLemmatizer()


# Load the resources globally
vectorizer, model, emotion_labels, stop_words, lemmatizer = load_artifacts()

# --- Pre-compile Regex (outside of function for efficiency) ---
emoji_pattern = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002500-\U00002BEF\U00002702-\U000027B0\U000024C2-\U0001F251]+",
    flags=re.UNICODE
)

# --- 2. Custom Preprocessing Function (UPDATED: now accepts NLTK args) ---
def preprocess(text, stop_words, lemmatizer, emoji_pattern):
    """
    Cleans and preprocesses text using the same logic as the training notebook.
    Takes NLTK objects as explicit arguments for safer execution.
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Ensure text is a string
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
    
    # 1. Preprocess the input text (Now calls with necessary globals)
    clean_text = preprocess(text, stop_words, lemmatizer, emoji_pattern)
    
    # Check for empty clean text after processing
    if not clean_text:
        st.warning("Processed text is empty. Analysis cannot be performed.")
        return pd.DataFrame()
    
    # 2. Vectorize the clean text
    text_vectorized = vectorizer.transform([clean_text])
    
    # 3. Get raw prediction (0 or 1 for each label)
    prediction_raw = model.predict(text_vectorized)[0]
    
    # 4. Get probability estimates (from the underlying classifier)
    all_probas = []
    if hasattr(model, 'estimators_'):
        for i, estimator in enumerate(model.estimators_):
            # L-R for multi-label usually provides probabilities from 0 to 1
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
            - Length of labels: {len_labels}
            - Length of Model Predictions: {len_predictions}
            
            **Action Required:** Ensure your emotion labels match your model's output.
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


# --- 4. Streamlit UI Design (Emotion Analysis) ---

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
    <div  class="caption-intro">
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
# Text input widget
user_input = st.text_area(
    "Enter Text Here:",
    placeholder="e.g., Hello church! God is good😊",
    height=150
)

st.markdown(
    """
    <script>
    // Apply custom CSS class to Streamlit textarea
    var textareas = window.parent.document.querySelectorAll('textarea');
    if (textareas.length > 0) {
        textareas[textareas.length - 1].classList.add('custom-textarea');
    }
    </script>
    """,
    unsafe_allow_html=True
)

# 2. Add an Analyze Button to trigger the check
if st.button("Check Input"):
    
    # 3. Check for Empty Input
    if not user_input.strip():
        st.warning("Input is empty. Please enter some text.")
    
    # 4. Check if the input contains ONLY numeric characters
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
        
        if prediction_results is not None and not prediction_results.empty:
            # --- Display Results ---
            st.subheader("Analysis Complete")
            
            # Display the primary predicted emotions
            predicted_emotions = prediction_results[prediction_results['Predicted'] == 1]['Emotion'].tolist()

            if predicted_emotions:
                st.success(f"**Primary Emotions Detected:** {', '.join([e.title() for e in predicted_emotions])}")
            else:
                st.info("No strong single emotion predicted. Showing top confidence scores.")


            col1, col2 = st.columns([2, 1.5])
            with col1:

                # Display the full confidence breakdown
                st.markdown("#### Confidence Breakdown")
                st.dataframe(
                prediction_results.drop(columns=['Predicted']),
                hide_index=True,
                width=350,
                height=170
                )
            with col2:
                # Optional: Display a bar chart of the top 10 confidences
                top_confidences = prediction_results.head(10)
                st.bar_chart(
                   top_confidences,
                   x='Emotion',
                   y='Confidence (%)',
                   color="#E6652B" 
                )
            # --- Display Interventions ---
            st.markdown(""" 
                        <div class="inter">
                             Intervention Suggestions

                        </div> """,
                        unsafe_allow_html=True
                        )
            display_interventions(prediction_results)

    else:
        st.warning("Please enter some text to analyze.")

#--- VOICE PROCESSING FUNCTIONS ---


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


# --- Sidebar UI (Voice Recognition) ---


# --- Voice Recording and Analysis (NEW SECTION, Aligned Left) ---
st.sidebar.subheader("🎙️ Voice Analyzer (STT)")
st.sidebar.markdown(
    """
    <p class="text">
    Use your microphone to record speech for text analysis.
    </p>
    """, unsafe_allow_html=True
)

voice_results_placeholder = st.sidebar.empty()

# --- Initialization and Global Configuration ---
# Must be defined outside of any Streamlit block (like st.sidebar or st.button)


RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {
            "urls": ["turn:global.xirsys.net:3478?transport=udp"],
            "username": "Onguka",
            "credential": "John@3567",
        }
    ],
    "iceCandidatePoolSize": 2, 
}

# Initialize session state for recorded audio data ---
if 'recorded_audio_data' not in st.session_state:
    st.session_state['recorded_audio_data'] = None

# --- UI for Voice Recording ---
with st.sidebar:
    voice_results_placeholder = st.empty() # Placeholder for results
    
    # 1. Create an empty element to display connection status
    st_webrtc_status = st.empty()
    st_webrtc_status.info("🎤 Waiting for microphone...")
    

    ctx = webrtc_streamer(
        key="speech_emotion_detector",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioAnalysisProcessor,
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True,
        rtc_configuration=RTC_CONFIGURATION,

    )

    if ctx.state.playing:
        st_webrtc_status.success("🟢 **Recording/Streaming** (Click **Stop** on the widget to pause)")
    else:
        st_webrtc_status.info("🎤 Click the **Start** button above to begin recording.")



# --- Logic to Save Audio Frames to Session State (Correct) ---
# Runs when stream is stopped and frames are available.

if not ctx.state.playing and ctx.audio_processor and ctx.audio_processor.audio_frames:
    
    # Check that we actually have audio data before saving
    if len(ctx.audio_processor.audio_frames) > 0:
        raw_audio_data = np.concatenate(ctx.audio_processor.audio_frames, axis=0)
        wav_bytes = raw_pcm_to_wav_bytes(raw_audio_data)
        
        st.session_state['recorded_audio_data'] = wav_bytes
        ctx.audio_processor.audio_frames.clear()
        
        st.sidebar.success("✅ Recording complete. Ready for analysis.")
    else:
        # Clear frames even if empty, but don't save to state
        ctx.audio_processor.audio_frames.clear()




# Now relies entirely on the saved session state data.

if st.sidebar.button("Analyze Recorded Voice", use_container_width=True, key="voice_analyze_btn"):
    
    # Clear previous results
    voice_results_placeholder.empty()

    # --- FIX: Check for saved data instead of raw frames ---
    wav_bytes_to_analyze = st.session_state.get('recorded_audio_data')

    if wav_bytes_to_analyze is None:
        # Warning if no audio was saved
        voice_results_placeholder.warning("Please record your voice and click the STOP button on the widget before analyzing.")
        
    else:
        with voice_results_placeholder.container():
            with st.spinner("Transcribing and analyzing text..."):
                
                # Use the saved WAV bytes directly for transcription
                transcribed_text = transcribe_audio(wav_bytes_to_analyze)
                
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
                    
                    #prediction of the trascribed text
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





# --- Data Preprocessing UI Functions (The Fix for NameError) ---

def ui_handle_missing_values(df_key):
    """
    UI and logic for handling missing values in the DataFrame.
    """
    df = st.session_state[df_key]
    st.subheader("1. Handle Missing Values")
    
    # Show initial state
    missing_summary = df.isnull().sum()
    missing_data = missing_summary[missing_summary > 0]
    
    if missing_data.empty:
        st.success("No missing values found in the DataFrame!")
    else:
        st.warning(f"Found missing values in {len(missing_data)} column(s).")
        st.dataframe(missing_data.rename("Missing Count"))
        
        st.markdown("#### Select Strategy")
        
        col_to_clean = st.selectbox(
            "Select Column to Handle Missing Values (Only columns with NaNs shown):",
            options=["-- Select a Column --"] + missing_data.index.tolist()
        )
        
        if col_to_clean != "-- Select a Column --":
            strategy = st.radio(
                f"Choose a strategy for **{col_to_clean}**:",
                options=["Drop Rows", "Fill with a value (e.g., 'unknown')", "Fill with Mode/Mean/Median (for numeric/categorical)"]
            )
            
            if st.button("Apply Missing Value Strategy"):
                new_df = df.copy() # Work on a copy
                
                if strategy == "Drop Rows":
                    new_df.dropna(subset=[col_to_clean], inplace=True)
                    st.session_state[df_key] = new_df
                    st.success(f"Successfully dropped rows with missing values in **{col_to_clean}**.")
                    st.dataframe(new_df.head())
                    
                elif strategy == "Fill with a value (e.g., 'unknown')":
                    fill_value = st.text_input("Enter value to fill NaNs with:", value="unknown")
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
    st.subheader("2. Text Cleaning & Normalization")
    st.markdown("Applies the full preprocessing pipeline: remove emojis, HTML, numbers, lowercase, remove punctuation, lemmatization, and stop word removal.")
    
    # Identify text columns (simplistic check)
    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if not text_columns:
        st.error("No suitable text columns (object dtype) found in the DataFrame.")
        return
    
    col_to_clean = st.selectbox(
        "Select Text Column to Clean:",
        options=text_columns
    )
    
    # Option to create a new column or overwrite
    cleaning_action = st.radio(
        "Cleaning Action:",
        options=["Create New Column", "Overwrite Existing Column"],
        key='cleaning_action_radio_2' # Changed key to avoid conflict
    )
    
    new_col_name = col_to_clean + "_clean"
    if cleaning_action == "Create New Column":
        new_col_name = st.text_input("New Column Name:", value=new_col_name, key='new_col_name_input')

    # Access the global NLTK components
    global stop_words, lemmatizer, emoji_pattern

    if st.button(f"Apply Preprocessing to **{col_to_clean}**"):
        try:
            with st.spinner(f"Cleaning text in column **{col_to_clean}**... This may take a moment."):
                
                # Apply the preprocessing function (passing the required global artifacts explicitly)
                # NOTE: This call is now safe because preprocess takes 4 arguments
                processed_series = df[col_to_clean].apply(
                    lambda x: preprocess(x, stop_words, lemmatizer, emoji_pattern)
                )

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
                st.dataframe(st.session_state[df_key][[col_to_clean, new_col_name]].head(10))

        except Exception as e:
            st.error(f"An error occurred during cleaning: {e}")


# --- Data Preprocessing UI (New Clickable and Left-Aligned Structure) ---

# Use an expander for the "Clickable Title" functionality
with st.expander(""" <h2 class="title">CSV Text Data Preprocessor Tool</h2> """, expanded=False,unsafe_allow_html=True):
    st.markdown(
        '<div class="caption-intro">Upload your CSV file and select a step from the navigation bar below to begin.</div>',
        unsafe_allow_html=True
    )

    # --- File Uploader ---
    uploaded_file = st.file_uploader("Choose a CSV file...", type="csv", key='preprocessor_uploader')

    if uploaded_file is not None:
        # 1. Initialization and Data Storage
        df_key = 'current_df'
        
        # Check if a new file was uploaded or if the session is fresh
        if df_key not in st.session_state or st.session_state['uploaded_file_name'] != uploaded_file.name:
            st.session_state[df_key] = pd.read_csv(uploaded_file)
            st.session_state['uploaded_file_name'] = uploaded_file.name
            st.success(f"CSV file '{uploaded_file.name}' loaded successfully! Select a step to begin.")
            
        df = st.session_state[df_key]
        
        st.write("---")

        # --- Left-Aligned Navigation and Content Area ---
        col_nav, col_content = st.columns([1, 3])

        with col_nav:
            st.subheader("🛠️ Steps")
            # Use st.radio for simple left-aligned navigation
            processing_step = st.radio(
                "Select a task:",
                options=[
                    "View Raw Data",
                    "Handle Missing Values",  
                    "Text Cleaning & Normalization"
                ],
                key='processing_step_select_main'
            )
            st.markdown("---")
            st.download_button(
                label="⬇️ Download Cleaned CSV",
                data=StringIO(st.session_state[df_key].to_csv(index=False)).getvalue(),
                file_name=f"cleaned_{st.session_state['uploaded_file_name']}",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_content:
            # --- Main Content Area Logic ---
            if processing_step == "View Raw Data":
                st.subheader(" Current Data View")
                st.info(f"Loaded DataFrame has {df.shape[0]} rows and {df.shape[1]} columns.")
                st.dataframe(df.head(20))
                
            elif processing_step == "Handle Missing Values":
                ui_handle_missing_values(df_key)
                
            elif processing_step == "Text Cleaning & Normalization":
                ui_clean_text_data(df_key)

            
    else:
        st.info("Awaiting CSV file upload.")


# Add a footer/info section
st.markdown("---")
st.markdown(
    "This model is a multi-label classifier trained on the GoEmotions dataset."
    
)