import os
import tempfile
import shutil
import base64
import mimetypes
from typing import List

from fastapi import FastAPI, UploadFile, File, Form

from config import client  # your Gemini/OpenAI client

app = FastAPI(title="AI-Powered Dental API")

# -------------------
# Uploads Directory
# -------------------
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "dental_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# -------------------
# Analyze Endpoint
# -------------------
@app.post("/analyze")
async def analyze(
    symptoms: str = Form(""),
    files: List[UploadFile] = File(None)
):
    try:
        saved_files = []
        user_content = [{"type": "text", "text": f"Patient symptoms: {symptoms}"}]

        if files:
            for file in files:
                file_path = os.path.join(UPLOAD_DIR, file.filename)

                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                mime_type, _ = mimetypes.guess_type(file_path)
                if not mime_type:
                    mime_type = "image/jpeg"

                with open(file_path, "rb") as f:
                    b64_image = base64.b64encode(f.read()).decode("utf-8")

                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
                })

                saved_files.append(file_path)

        if not saved_files:
            return {"error": "Please upload at least one image."}

        response = await client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": "You are a dental AI assistant. Summarize dental issues from photos."},
                {"role": "user", "content": user_content}
            ]
        )

        return {
            "saved_files": saved_files,
            "symptoms": symptoms,
            "analysis": response.choices[0].message.content   # ✅ fixed
        }

    except Exception as e:
        return {"error": str(e)}
