import streamlit as st
import nest_asyncio
nest_asyncio.apply()

from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor

import speech_recognition as sr
from gtts import gTTS
import os
import base64
import time

# --- Tool Functions & Agent Setup ---

def setup_qa_chain(transcript):
    """Creates a RAG chain for a given transcript."""
    if not transcript:
        return None
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(transcript)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", convert_system_message_to_human=True)
        retriever = vector_store.as_retriever()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False # We only need the answer for the agent
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error creating RAG chain: {e}")
        return None

def create_agent(qa_chain):
    """Creates an AI agent with tools."""
    if 'agent_executor' in st.session_state and st.session_state.agent_executor:
        return st.session_state.agent_executor

    # Tool 1: The RAG chain to query the YouTube transcript
    youtube_tool = Tool(
        name="query_youtube_transcript",
        func=lambda query: qa_chain({"query": query})["result"],
        description="Use this tool to answer questions about the content of the processed YouTube video transcript. Input should be a specific question."
    )

    # Tool 2: A general web search tool
    search_tool = TavilySearchResults(max_results=2)

    tools = [youtube_tool, search_tool]
    
    # Get the pre-built agent prompt
    prompt = hub.pull("hwchase17/xml-agent-convo")
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    
    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor


# --- Helper Functions (Audio, Cleanup, etc.) ---

def cleanup_audio_files():
    for filename in os.listdir('.'):
        if filename.endswith(".mp3"):
            try:
                os.remove(filename)
            except Exception as e:
                print(f"Error deleting file {filename}: {e}")

def get_transcript(youtube_url):
    try:
        video_id = youtube_url.split("v=")[1].split("&")[0]
        api = YouTubeTranscriptApi()
        transcript_data = api.fetch(video_id, languages=['en', 'hi'])
        return " ".join([snippet.text for snippet in transcript_data])
    except Exception as e:
        st.error(f"Error getting transcript: {e}")
        return None

def get_speech_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            text = r.recognize_google(audio)
            st.success("Processing your question...")
            return text
        except Exception:
            st.warning("Could not understand audio. Please try again.")
            return None

def text_to_speech(text):
    try:
        audio_file = f"response_{int(time.time())}.mp3"
        tts = gTTS(text=text, lang='en')
        tts.save(audio_file)
        return audio_file
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")
        return None

def autoplay_audio(file_path: str):
    if not os.path.exists(file_path):
        return
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f'<audio controls autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        st.markdown(md, unsafe_allow_html=True)

# --- Streamlit App ---

st.set_page_config(page_title="YouTube AI Agent", layout="centered")
st.title("🤖 YouTube Video AI Agent 📺")

with st.sidebar:
    st.header("Configuration")
    google_api_key = st.text_input("Google API Key", type="password")
    tavily_api_key = st.text_input("Tavily API Key", type="password")
    
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    if tavily_api_key:
        os.environ["TAVILY_API_KEY"] = tavily_api_key

    st.info("Enter a YouTube URL to create an agent that can answer questions about it.")

# Initialize session state
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "messages" not in st.session_state:
    st.session_state.messages = []

youtube_url = st.text_input("Enter YouTube URL:", key="youtube_url")

if st.button("Create Agent", key="create_agent"):
    cleanup_audio_files()
    if youtube_url and google_api_key and tavily_api_key:
        with st.spinner("Processing video and creating agent..."):
            transcript = get_transcript(youtube_url)
            if transcript:
                qa_chain = setup_qa_chain(transcript)
                if qa_chain:
                    st.session_state.agent_executor = create_agent(qa_chain)
                    st.session_state.messages = []
                    st.success("Agent created! It can now answer questions about the video and search the web.")
    else:
        st.warning("Please enter a YouTube URL and all required API keys in the sidebar.")

# Chat logic
if st.session_state.agent_executor:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "audio" in message and message["audio"]:
                autoplay_audio(message["audio"])

    prompt = None
    mic_available = True
    try:
        with sr.Microphone() as source: pass
    except OSError:
        mic_available = False

    if mic_available:
        _, col2 = st.columns([0.85, 0.15])
        with col2:
            if st.button("Ask 🎤"):
                prompt = get_speech_input()

    text_prompt = st.chat_input("Ask a question...")
    if text_prompt:
        prompt = text_prompt

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent_executor.invoke({"input": prompt})
                result = response["output"]
                st.markdown(result)
                
                audio_file = text_to_speech(result)
                if audio_file:
                    autoplay_audio(audio_file)
                
                st.session_state.messages.append({"role": "assistant", "content": result, "audio": audio_file})
        st.rerun()
else:
    st.info("Process a YouTube video to start the chat.")
