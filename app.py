import streamlit as st
# --- ADDED: Fix for asyncio event loop error ---
import nest_asyncio
nest_asyncio.apply()
# ---------------------------------------------
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
# --- Imports for Google Generative AI ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import speech_recognition as sr
from gtts import gTTS
import os
import base64
import time

# --- Helper Functions ---

def cleanup_audio_files():
    """Deletes all .mp3 files in the current directory."""
    for filename in os.listdir('.'):
        if filename.endswith(".mp3"):
            try:
                os.remove(filename)
            except Exception as e:
                # Log error to console instead of UI to avoid clutter
                print(f"Error deleting file {filename}: {e}")

def get_transcript(youtube_url):
    """Extracts the transcript from a YouTube video using the fetch method."""
    try:
        video_id = youtube_url.split("v=")[1].split("&")[0]
        
        api = YouTubeTranscriptApi()
        transcript_data = api.fetch(video_id, languages=['en', 'hi'])
        
        transcript_text = " ".join([snippet.text for snippet in transcript_data])
        return transcript_text
    except Exception as e:
        if "No transcript found for any of the requested languages" in str(e):
            try:
                video_id = youtube_url.split("v=")[1].split("&")[0]
                st.warning("Could not find a transcript in English or Hindi. Using the first available transcript.")
                api = YouTubeTranscriptApi()
                transcript_data = api.fetch(video_id)
                transcript_text = " ".join([snippet.text for snippet in transcript_data])
                return transcript_text
            except Exception as fallback_e:
                st.error(f"Error getting transcript: {fallback_e}")
                return None
        else:
            st.error(f"Error getting transcript: {e}")
            return None


def create_vector_store(text):
    """Creates a FAISS vector store from the given text."""
    if not text:
        return None
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_qa_chain(vector_store):
    """Creates a RetrievalQA chain using Gemini."""
    if vector_store is None:
        return None
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", convert_system_message_to_human=True)
        retriever = vector_store.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        return None

def get_speech_input():
    """Gets audio input from the user and converts it to text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            text = r.recognize_google(audio)
            st.success("Processing your question...")
            return text
        except sr.WaitTimeoutError:
            st.warning("Listening timed out. Please try again.")
            return None
        except sr.UnknownValueError:
            st.warning("Could not understand audio. Please try again.")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition; {e}")
            return None


def text_to_speech(text):
    """Converts text to speech and returns the audio file path."""
    try:
        # Use a unique filename to prevent browser caching issues
        audio_file = f"response_{int(time.time())}.mp3"
        tts = gTTS(text=text, lang='en')
        tts.save(audio_file)
        return audio_file
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")
        return None

def autoplay_audio(file_path: str):
    """Plays the audio file automatically."""
    # Check if the file exists before trying to open it
    if not os.path.exists(file_path):
        return
        
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

# --- Streamlit App ---

st.set_page_config(page_title="YouTube Voice Chatbot", layout="centered")

st.title("🗣️ YouTube Video Voice Chatbot 📺")

# --- API Key Management in Sidebar ---
with st.sidebar:
    st.header("Configuration")
    google_api_key = st.text_input("Google API Key", type="password")
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key

    st.info("Enter a YouTube URL in the main panel to begin.")

# --- Main App Logic ---

# Initialize session state variables
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

youtube_url = st.text_input("Enter YouTube URL:", key="youtube_url")

if st.button("Process Video", key="process_video"):
    # --- ADDED: Cleanup old audio files when processing a new video ---
    cleanup_audio_files()

    if youtube_url and google_api_key:
        with st.spinner("Processing video... This may take a moment."):
            transcript = get_transcript(youtube_url)
            if transcript:
                vector_store = create_vector_store(transcript)
                if vector_store:
                    st.session_state.qa_chain = get_qa_chain(vector_store)
                    st.session_state.messages = [] # Clear previous chat
                    st.success("Video processed! You can now ask questions below.")
    elif not google_api_key:
        st.warning("Please enter your Google API Key in the sidebar.")
    else:
        st.warning("Please enter a YouTube URL.")

# Display chat messages from history
if st.session_state.qa_chain:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "audio" in message and message["audio"]:
                autoplay_audio(message["audio"])

    # --- CHAT INPUT AREA ---
    prompt = None
    
    # Create columns to position the voice button to the right
    col1, col2 = st.columns([0.85, 0.15])
    with col2:
        if st.button("Ask with Voice 🎤"):
            prompt = get_speech_input()

    # Get text input from the user
    text_prompt = st.chat_input("Ask a question about the video...")
    if text_prompt:
        prompt = text_prompt

    # If a prompt was received (from either voice or text), process it
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain({"query": prompt})
                result = response["result"]
                st.markdown(result)
                
                audio_file = text_to_speech(result)
                if audio_file:
                    autoplay_audio(audio_file)
                
                st.session_state.messages.append({"role": "assistant", "content": result, "audio": audio_file})
        
        # Rerun to clear the input fields and update the chat display
        st.rerun()

else:
    st.info("Process a YouTube video to start the chat.")
