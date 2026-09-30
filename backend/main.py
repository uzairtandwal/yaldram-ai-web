from fastapi import FastAPI, UploadFile, File, Form
import tempfile
import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from rag import ask_question

app = FastAPI(title="Islamic Q&A Chatbot API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    sessionId: str = None
    audioDuration: int = 0

class SourceItem(BaseModel):
    title: str
    url: str

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]

@app.post("/api/chat", response_model=AskResponse)
@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    # Call the RAG logic
    result = ask_question(req.question)
    return result


@app.post("/api/voice-message", response_model=AskResponse)
async def voice_endpoint(file: UploadFile = File(...), audioDuration: int = Form(0)):
    # 1. Save audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(await file.read())
        temp_audio_path = temp_audio.name
        
    try:
        
        print("Transcribing audio...")
        with open(temp_audio_path, "rb") as f:
            audio_bytes = f.read()
        transcribe_model = genai.GenerativeModel("gemini-3.5-flash")
        
        prompt = "Transcribe this audio exactly as spoken in Roman Urdu or Urdu. Return ONLY the transcribed text without any extra explanation or quotes."
        transcription_response = transcribe_model.generate_content([
            prompt,
            {"mime_type": "audio/webm", "data": audio_bytes}
        ])
        transcribed_text = transcription_response.text.strip()

        
        print(f"User asked (via voice): {transcribed_text}")
        
        # 3. Call normal RAG pipeline
        result = ask_question(transcribed_text)
        return result
    except Exception as e:
        print("Voice error:", e)
        return {"answer": f"Sorry, I couldn't process your voice message. Error: {str(e)}", "sources": []}
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


# Serve the static frontend
FRONTEND_DIR = r"C:\Users\Uzair Ali\Documents\copy yaldram\yaldram_ai"

# Mount the assets directory specifically
app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

# Fallback for any other unmatched route (React SPA)
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    # Check if the requested path is an actual file in FRONTEND_DIR
    file_path = os.path.join(FRONTEND_DIR, full_path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    # Otherwise, return index.html for React Router
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))



