import os
import gradio as gr
from groq import Groq
from gtts import gTTS
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

def voice_assistant(audio_path):
    if audio_path is None:
        return "No audio detected.", "", None

    # 1. Speech-to-Text (STT) via Groq Whisper
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(audio_path, audio_file.read()),
            model="whisper-large-v3-turbo",
            response_format="text"
        )
    user_text = transcription.strip()

    # 2. LLM Response via Groq
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful voice assistant. Keep your answers clear and concise."
            },
            {
                "role": "user",
                "content": user_text,
            }
        ],
        model="openai/gpt-oss-20b",
    )
    ai_reply = chat_completion.choices[0].message.content

    # 3. Text-to-Speech (TTS) via gTTS
    output_audio_path = "response.mp3"
    tts = gTTS(text=ai_reply, lang="en")
    tts.save(output_audio_path)

    return user_text, ai_reply, output_audio_path

# 4. Launch Gradio Interface
interface = gr.Interface(
    fn=voice_assistant,
    inputs=gr.Audio(sources=["microphone"], type="filepath", label="Speak here"),
    outputs=[
        gr.Textbox(label="You said (STT)"),
        gr.Textbox(label="AI Reply (LLM)"),
        gr.Audio(label="AI Voice (TTS)", autoplay=True)
    ],
    title="Free Voice AI Assistant (Groq + gTTS)"
)

if __name__ == "__main__":
    interface.launch()