import os
import io
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from groq import Groq
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# UI file
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# Handle Voice -> STT -> LLM -> TTS
@app.post("/api/voice")
async def process_voice(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()

    # 1. Speech-to-Text (Groq Whisper)
    transcription = client.audio.transcriptions.create(
        file=("recording.webm", audio_bytes),
        model="whisper-large-v3-turbo",
        response_format="text"
    )
    user_text = transcription.strip()

    if not user_text:
        return JSONResponse({"user_text": "", "ai_reply": "I couldn't hear anything.", "audio_base64": None})

    # 2. LLM Response (Groq)
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful voice assistant. Keep your answers clear and concise."},
            {"role": "user", "content": user_text}
        ],
        model="openai/gpt-oss-20b",
    )
    ai_reply = chat_completion.choices[0].message.content

    # 3. Text-to-Speech (gTTS in memory)
    mp3_fp = io.BytesIO()
    tts = gTTS(text=ai_reply, lang="en")
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    audio_base64 = base64.b64encode(mp3_fp.read()).decode("utf-8")

    return JSONResponse({
        "user_text": user_text,
        "ai_reply": ai_reply,
        "audio_base64": audio_base64
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)