# -*- coding: utf-8 -*-
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn
import shutil
import json
from datetime import datetime
import numpy as np

from api.config import UPLOAD_DIR
from api.classifier import get_classifier

def fix_encoding(text):
    if not isinstance(text, str):
        return text
    try:
        if 'Ã©' in text or 'Ã' in text:
            text = text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return text

def convert_to_json_serializable(obj):
    if isinstance(obj, dict):
        return {fix_encoding(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, str):
        return fix_encoding(obj)
    elif isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

app = FastAPI(title="Document Classification API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

web_dir = Path(__file__).parent.parent / "web"
static_dir = web_dir / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

classifier = None
_classifier_lock = False

def get_classifier_instance():
    global classifier, _classifier_lock
    if classifier is None and not _classifier_lock:
        _classifier_lock = True
        try:
            use_cpu = False
            classifier = get_classifier(use_cpu=use_cpu)
            print("Classifieur initialisé, modèles chargés")
        except Exception as e:
            print(f"Erreur lors du chargement des modèles: {e}")
            _classifier_lock = False
            raise
    return classifier

@app.get("/", response_class=HTMLResponse)
async def root():
    web_file = web_dir / "index.html"
    if web_file.exists():
        return web_file.read_text(encoding='utf-8')
    return HTMLResponse(content="<html><body><h1>Document Classification API</h1><p>API is ready. Use /api endpoints or open web/index.html</p></body></html>")

@app.get("/result.html", response_class=HTMLResponse)
async def result_page():
    web_file = web_dir / "result.html"
    if web_file.exists():
        return web_file.read_text(encoding='utf-8')
    return HTMLResponse(content="<html><body><h1>Results page not found</h1></body></html>")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": {
            "vlm": classifier.qwen_model is not None if classifier else False,
            "orb": classifier.orb_gallery is not None if classifier else False,
            "roberta": classifier.roberta_model is not None if classifier else False
        }
    }

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_ext = Path(file.filename).suffix.lower()
        is_pdf = file_ext == '.pdf'
        allowed_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.pdf'}
        
        if file_ext not in allowed_exts:
            raise HTTPException(
                status_code=400,
                detail=f"Format non supporté. Formats acceptés: {', '.join(allowed_exts)}"
            )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "filename": safe_filename,
            "file_id": safe_filename,
            "is_pdf": is_pdf,
            "size": file_path.stat().st_size
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur upload: {str(e)}")

@app.api_route("/api/classify", methods=["GET", "POST"])
async def classify_document(file_id: str):
    try:
        file_path = UPLOAD_DIR / file_id
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Fichier non trouvé")
        
        is_pdf = file_path.suffix.lower() == '.pdf'
        
        classifier_instance = get_classifier_instance()
        results = classifier_instance.classify_document(file_path, is_pdf=is_pdf)
        
        results["file_id"] = file_id
        results["is_pdf"] = is_pdf
        
        results = convert_to_json_serializable(results)
        
        return JSONResponse(
            content=results,
            media_type="application/json; charset=utf-8"
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur classification: {str(e)}")

@app.post("/api/classify/upload")
async def classify_upload(file: UploadFile = File(...)):
    try:
        upload_result = await upload_file(file)
        file_id = upload_result["file_id"]
        
        classify_result = await classify_document(file_id)
        
        return classify_result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/file/{file_id}")
async def get_file(file_id: str):
    try:
        file_path = UPLOAD_DIR / file_id
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Fichier non trouvé")
        return FileResponse(file_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/debug/{file_id}")
async def debug_classification(file_id: str):
    try:
        file_path = UPLOAD_DIR / file_id
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Fichier non trouvé")
        
        is_pdf = file_path.suffix.lower() == '.pdf'
        results = classifier.classify_document(file_path, is_pdf=is_pdf)
        
        results["debug"] = {
            "device": str(classifier.device),
            "models_loaded": {
                "vlm": classifier.qwen_model is not None,
                "orb": classifier.orb_gallery is not None,
                "roberta": classifier.roberta_model is not None
            }
        }
        
        return JSONResponse(content=results)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
