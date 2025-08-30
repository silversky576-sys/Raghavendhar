import streamlit as st
import os
import io
import requests
from dotenv import load_dotenv
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import uuid
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# IBM Watsonx and TTS configuration
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com/ml/v1-beta/generation/text")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
TTS_API_KEY = os.getenv("TTS_API_KEY")
TTS_URL = os.getenv("TTS_URL", "https://api.us-south.text-to-speech.watson.cloud.ibm.com")

# Initialize IBM Watson TTS
authenticator = IAMAuthenticator(TTS_API_KEY)
tts = TextToSpeechV1(authenticator=authenticator)
tts.set_service_url(TTS_URL)

# Function to rewrite text with Watsonx Granite model
def rewrite_text(text, tone):
    headers = {
        "Authorization": f"Bearer {WATSONX_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"Rewrite the following text in a {tone} tone while preserving the original meaning:\n\n{text}"
    data = {
        "model_id": "ibm/granite-13b-chat-v2",
        "input": prompt,
        "parameters": {
            "max_new_tokens": 1000,
            "temperature": 0.7,
            "stop_sequences": []
        },
        "project_id": WATSONX_PROJECT_ID
    }
    try:
        response = requests.post(WATSONX_URL, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result.get("results", [{}])[0].get("generated_text", text)
    except requests.exceptions.RequestException as e:
        st.error(f"Error rewriting text: {e}")
        return text

# Function to generate audio from text
def generate_audio(text, voice):
    try:
        audio = tts.synthesize(
            text,
            voice=voice,
            accept="audio/mp3"
        ).get_result().content
        return audio
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

# Initialize session state for past narrations
if "narrations" not in st.session_state:
    st.session_state.narrations = []

# Streamlit UI
st.title("EchoVerse: AI-Powered Audiobook Creator")

# Input section
st.header("Input Text")
input_method = st.radio("Choose input method:", ("Paste Text", "Upload File"))
text_input = ""
if input_method == "Paste Text":
    text_input = st.text_area("Paste your text here:", height=200)
else:
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    if uploaded_file:
        text_input = uploaded_file.read().decode("utf-8")

# Tone and voice selection
st.header("Customize Narration")
col1, col2 = st.columns(2)
with col1:
    tone = st.selectbox("Select Tone:", ["Neutral", "Suspenseful", "Inspiring"])
with col2:
    voice = st.selectbox("Select Voice:", ["en-US_LisaV3Voice", "en-US_MichaelV3Voice", "en-US_AllisonV3Voice"])

# Process button
if st.button("Generate Narration", disabled=not text_input):
    if text_input:
        with st.spinner("Rewriting text..."):
            rewritten_text = rewrite_text(text_input, tone)
        with st.spinner("Generating audio..."):
            audio_data = generate_audio(rewritten_text, voice)
        
        if audio_data:
            # Generate unique ID and timestamp for narration
            narration_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save to session state
            st.session_state.narrations.append({
                "id": narration_id,
                "original_text": text_input,
                "rewritten_text": rewritten_text,
                "audio": audio_data,
                "tone": tone,
                "voice": voice,
                "timestamp": timestamp
            })
            
            # Display side-by-side text
            st.header("Text Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Text")
                st.write(text_input)
            with col2:
                st.subheader(f"Rewritten Text ({tone})")
                st.write(rewritten_text)
            
            # Audio playback
            st.header("Narration")
            st.audio(audio_data, format="audio/mp3")
            
            # Download button
            st.download_button(
                label="Download Audio",
                data=audio_data,
                file_name=f"narration_{timestamp}.mp3",
                mime="audio/mp3"
            )
    else:
        st.warning("Please provide text input.")

# Past Narrations panel
st.header("Past Narrations")
if st.session_state.narrations:
    for narration in reversed(st.session_state.narrations):
        with st.expander(f"Narration from {narration['timestamp']} (Tone: {narration['tone']}, Voice: {narration['voice']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Text")
                st.write(narration["original_text"])
            with col2:
                st.subheader(f"Rewritten Text ({narration['tone']})")
                st.write(narration["rewritten_text"])
            st.audio(narration["audio"], format="audio/mp3")
            st.download_button(
                label="Download Audio",
                data=narration["audio"],
                file_name=f"narration_{narration['timestamp']}.mp3",
                mime="audio/mp3"
            )
else:
    st.info("No narrations generated in this session.")

# Instructions for running the app
st.markdown("""
### Setup Instructions
1. Install dependencies: `pip install streamlit python-dotenv ibm-watson requests`
2. Create a `.env` file with:
   ```
   WATSONX_API_KEY=your_watsonx_api_key
   WATSONX_PROJECT_ID=your_watsonx_project_id
   TTS_API_KEY=your_tts_api_key
   ```
3. Run the app: `streamlit run app.py`
""")