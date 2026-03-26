import os
import uuid
from fastapi import UploadFile, HTTPException
from app.core.config import settings

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".pdf"}

async def validate_and_save_file(file: UploadFile) -> str:
    """Validate file type/size and save to disk. Returns saved file path."""
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    contents = await file.read()
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(settings.UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
    with open(save_path, "wb") as f:
        f.write(contents)

    # Reset file pointer for any downstream reads
    await file.seek(0)
    return save_path
