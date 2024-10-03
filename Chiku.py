import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain
import os
import requests
import base64
from streamlit_lottie import st_lottie, st_lottie_spinner

# Custom CSS and page settings
st.set_page_config(
    page_title="Chiku",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Remove Streamlit top navigation bar and footer
st.markdown("""
    <style>
        /* Hide Streamlit top navigation and footer */
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

def load_lottieurl(url: str):
    """Fetch Lottie animation from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error("Failed to load animation. Please try again.")
        return None

# Load Lottie animations
lottie_url_hello = "https://lottie.host/9482fe3d-8952-4031-bde7-5e59a14de933/BGOjDWLh6U.json"
lottie_hello = load_lottieurl(lottie_url_hello)
st_lottie(lottie_hello, key="hello", quality="high", height=200)

# Apply custom CSS styles to remove the background from Lottie animations
st.markdown("""
    <style>
    body {
        font-family: 'Poppins', sans-serif;
        background-color: #000000;
        color: #e1e1e1;
    }
    h1 {
        font-size: 3rem;
        font-weight: bold;
        background: -webkit-linear-gradient(45deg, #00d4ff, #007bff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 20px;
        animation: float 3s infinite ease-in-out;
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    .main {
        background-color: #000000;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.5);
    }
    .small-gif {
        width: 250px;
        height: auto;
        display: block;
        margin: 0 auto;
    }
    .centered {
        display: flex;
        justify-content: center;
    }
    .divider {
        margin-top: 10px;
        margin-bottom: 10px;
        border-bottom: 1px solid #888;
    }
    .lottie {
        background: none; /* Removes Lottie background */
    }
    </style>
""", unsafe_allow_html=True)

# Load sensitive data from environment variables for security
try:
    HF_TOKEN = st.secrets["HUGGINGFACE_ACCESS_TOKEN"]
    XI_API_KEY = st.secrets["VOICE_API"]
    VOICE_ID = st.secrets["REAL_VOICE_ID"]
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
    os.environ["VOICE_API"] = XI_API_KEY
    os.environ["REAL_VOICE_ID"] = VOICE_ID

except KeyError:
    st.error("API token not found. Please check your credentials.")
    st.stop()

# Initialize the external language model (Hugging Face or another)
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
try:
    if not HF_TOKEN:
        raise ValueError("Missing API token.")
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=1024, temperature=0.7, token=HF_TOKEN)
except Exception as e:
    st.error("Error initializing the language model. Please check your credentials or model ID.")
    st.stop()

# Set up LangChain's prompt template
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Initialize LLMChain with the prompt and model
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Function to generate and play audio
def play_audio(text):
    """Generate and play audio from the response text."""
    if not XI_API_KEY:
        st.error("API key for text-to-speech is missing.")
        return

    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
    headers = {"Accept": "application/json", "xi-api-key": XI_API_KEY}
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.6,
            "similarity_boost": 0.8,
            "style": 1,
            "use_speaker_boost": True
        }
    }

    audio_file_path = "generated_audio.mp3"
    
    try:
        response = requests.post(tts_url, headers=headers, json=data, stream=True)
        response.raise_for_status()

        # Save the audio content locally
        with open(audio_file_path, 'wb') as audio_file:
            for chunk in response.iter_content(chunk_size=1024):
                audio_file.write(chunk)

        # Display GIF and play audio
        gif_file_path = "1.gif"
        try:
            gif_placeholder = st.empty()
            gif_placeholder.markdown(f"<div class='centered'><img src='data:image/gif;base64,{base64.b64encode(open(gif_file_path, 'rb').read()).decode()}' class='small-gif'/></div>", unsafe_allow_html=True)
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        except FileNotFoundError:
            st.error("GIF file not found.")
            return

        # Play audio using st.audio
        st.audio(audio_file_path, format="audio/mpeg", autoplay=True)

        # Hide the audio bar
        st.markdown("""
            <style>
            audio {
                display: none;
            }
            </style>
        """, unsafe_allow_html=True)

        # Remove the audio file after use to save storage
        os.remove(audio_file_path)

    except requests.exceptions.RequestException as e:
        st.error("An error occurred while generating audio. Please try again later.")

# Function to process the model response
def process_response(response):
    """Split the response into text and code sections."""
    code_blocks = []
    text_blocks = []
    in_code_block = False

    for line in response.split('\n'):
        if line.startswith("```"):
            in_code_block = not in_code_block
            if in_code_block:
                code_blocks.append("")
            continue
        if in_code_block:
            code_blocks[-1] += line + '\n'
        else:
            text_blocks.append(line)

    return "\n".join(text_blocks), "\n".join(code_blocks)

# Streamlit input form
with st.form(key='input_form', clear_on_submit=True):
    user_input = st.text_input("Enter your question:", key="user_input")
    submit_button = st.form_submit_button(label="Send")

# Load Lottie animations for loading states
lottie_loading_audio_url = "https://lottie.host/c042c660-bc89-42d4-aae2-88c5d3d5111d/bWZqfWBMWt.json"
lottie_audio = load_lottieurl(lottie_loading_audio_url)

lottie_url_download = "https://lottie.host/c24cac91-e6c5-499d-8701-12dada9986ba/Wewg2Y3VzZ.json"
lottie_download = load_lottieurl(lottie_url_download)

# Handle user input and model responses
if submit_button and user_input:
    try:
        with st_lottie_spinner(lottie_download, key="download", height=140, width=150, quality="high"):
            response = llm_chain.invoke(user_input)['text']
        
        # Process the response
        text_response, code_response = process_response(response)

        # Display the user input and model's text response
        st.code(user_input, language='markdown')
        st.write(text_response)

        if text_response:
            with st_lottie_spinner(lottie_audio, key="load_audio", height=140, width=150, quality="high"):
                play_audio(text_response)  # Play audio for the text part only

        # Display any code in the response
        if code_response:
            st.code(code_response, language='python')

    except Exception as e:
        st.error("An error occurred while processing your request. Please try again later.")