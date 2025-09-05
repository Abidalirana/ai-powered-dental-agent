import asyncio
import uuid
import os
import tempfile
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict

from app import run_dental_agent
from config import client

app = FastAPI(title="AI-Powered Dental API")

# Mount a static directory for serving uploaded files
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "dental_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")

# In-memory chat storage
chat_sessions: Dict[str, list] = {}

class AskRequest(BaseModel):
    query: str
    session_id: str | None = None

class AskResponse(BaseModel):
    session_id: str
    response: str


@app.post("/ask", response_model=AskResponse)
async def ask_dental_agent(request: AskRequest):
    """Ask dental assistant a question (per-session memory)."""
    session_id = request.session_id or str(uuid.uuid4())

    history = chat_sessions.get(session_id, [])
    history.append({"role": "user", "content": request.query})

    response = await run_dental_agent(request.query)

    history.append({"role": "assistant", "content": response})
    chat_sessions[session_id] = history

    return AskResponse(session_id=session_id, response=response)


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), symptoms: str = Form("")):
    """Analyze dental photo + symptoms (local URL)."""
    try:
        # Save uploaded file in temp folder
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Public URL served by FastAPI (works locally)
        image_url = f"http://127.0.0.1:8000/files/{file.filename}"

        # Call Gemini
        response = await client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": "You are a dental AI assistant. Analyze dental photos for possible issues."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Patient symptoms: {symptoms}"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]}
            ]
        )

        return {"analysis": response.choices[0].message["content"]}

    except Exception as e:
        return {"error": str(e)}
