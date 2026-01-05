# -*- coding: utf-8 -*-
"""
=============================================================================
SERVEUR API FASTAPI - CLASSIFICATION DE DOCUMENTS MAROCAINS
=============================================================================
Ce fichier est le point d'entrée principal de l'application.
Il expose une API REST pour :
- Upload de documents (images ou PDF)
- Classification automatique des documents
- Affichage des résultats via interface web

Démarrage du serveur:
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints disponibles:
    GET  /                  → Page d'accueil (upload)
    GET  /result.html       → Page des résultats
    GET  /health            → État de santé de l'API
    POST /api/upload        → Upload d'un fichier
    POST /api/classify      → Classification d'un document
    GET  /api/file/{id}     → Récupération d'un fichier uploadé

Auteur: Système de Classification de Documents Marocains
=============================================================================
"""

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

# Import de la configuration et du classifieur
from api.config import UPLOAD_DIR
from api.classifier import get_classifier


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def fix_encoding(text):
    """
    Corrige les problèmes d'encodage UTF-8.
    Certains textes peuvent être mal encodés (UTF-8 interprété comme Latin-1).
    
    Args:
        text: Texte à corriger
    
    Returns:
        Texte avec encodage corrigé
    """
    if not isinstance(text, str):
        return text
    try:
        # Détecte si le texte contient des caractères mal encodés
        if 'Ã©' in text or 'Ã' in text:
            # Ré-encode de Latin-1 vers UTF-8
            text = text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return text


def convert_to_json_serializable(obj):
    """
    Convertit récursivement un objet en format JSON sérialisable.
    Gère les types NumPy (float32, int64, ndarray) qui ne sont pas
    directement sérialisables en JSON.
    
    Args:
        obj: Objet à convertir (dict, list, numpy array, etc.)
    
    Returns:
        Objet converti en types Python natifs
    """
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


# =============================================================================
# INITIALISATION DE L'APPLICATION FASTAPI
# =============================================================================

# Création de l'application FastAPI avec métadonnées
app = FastAPI(
    title="Document Classification API",
    version="1.0.0",
    description="API de classification automatique de documents marocains"
)

# Configuration CORS pour permettre les requêtes cross-origin
# (nécessaire pour le frontend web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Autorise toutes les origines
    allow_credentials=True,
    allow_methods=["*"],        # Autorise toutes les méthodes HTTP
    allow_headers=["*"],        # Autorise tous les headers
)

# Montage des fichiers statiques (CSS, JS, images)
web_dir = Path(__file__).parent.parent / "web"
static_dir = web_dir / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# =============================================================================
# GESTION DU CLASSIFIEUR (LAZY LOADING)
# =============================================================================

# Instance globale du classifieur (chargée à la demande)
classifier = None
_classifier_lock = False  # Verrou pour éviter les chargements multiples


def get_classifier_instance():
    """
    Récupère l'instance du classifieur (pattern Singleton avec lazy loading).
    Le classifieur est chargé uniquement lors de la première requête
    pour accélérer le démarrage du serveur.
    
    Returns:
        Instance du DocumentClassifier avec tous les modèles chargés
    """
    global classifier, _classifier_lock
    
    if classifier is None and not _classifier_lock:
        _classifier_lock = True
        try:
            use_cpu = False  # Utilise GPU si disponible
            classifier = get_classifier(use_cpu=use_cpu)
            print("Classifieur initialisé, modèles chargés")
        except Exception as e:
            print(f"Erreur lors du chargement des modèles: {e}")
            _classifier_lock = False
            raise
    
    return classifier


# =============================================================================
# ENDPOINTS - PAGES WEB
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Page d'accueil de l'application.
    Sert le fichier index.html qui contient l'interface d'upload.
    """
    web_file = web_dir / "index.html"
    if web_file.exists():
        return web_file.read_text(encoding='utf-8')
    return HTMLResponse(
        content="<html><body><h1>Document Classification API</h1>"
                "<p>API is ready. Use /api endpoints or open web/index.html</p></body></html>"
    )


@app.get("/result.html", response_class=HTMLResponse)
async def result_page():
    """
    Page des résultats de classification.
    Affiche les détails de la classification après traitement.
    """
    web_file = web_dir / "result.html"
    if web_file.exists():
        return web_file.read_text(encoding='utf-8')
    return HTMLResponse(content="<html><body><h1>Results page not found</h1></body></html>")


# =============================================================================
# ENDPOINTS - API
# =============================================================================

@app.get("/health")
async def health():
    """
    Endpoint de santé pour vérifier l'état de l'API.
    Retourne l'état de chaque modèle (chargé ou non).
    
    Returns:
        JSON avec le statut et l'état des modèles
    """
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
    """
    Upload d'un fichier (image ou PDF).
    Le fichier est sauvegardé avec un timestamp pour éviter les conflits.
    
    Args:
        file: Fichier uploadé (multipart/form-data)
    
    Returns:
        JSON avec le file_id, filename, is_pdf, et size
    
    Raises:
        HTTPException 400: Format de fichier non supporté
        HTTPException 500: Erreur lors de l'upload
    """
    try:
        # Vérification de l'extension du fichier
        file_ext = Path(file.filename).suffix.lower()
        is_pdf = file_ext == '.pdf'
        allowed_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.pdf'}
        
        if file_ext not in allowed_exts:
            raise HTTPException(
                status_code=400,
                detail=f"Format non supporté. Formats acceptés: {', '.join(allowed_exts)}"
            )
        
        # Génération d'un nom de fichier unique avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        # Sauvegarde du fichier sur le disque
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
    """
    Classification d'un document uploadé.
    Utilise le pipeline multi-modal (VLM + ORB + RoBERTa + OCR).
    
    Args:
        file_id: Identifiant du fichier (retourné par /api/upload)
    
    Returns:
        JSON avec les résultats de classification:
        - final_level1: Classe principale (4 classes)
        - final_level2: Sous-classe détaillée (14 classes)
        - final_confidence: Score de confiance
        - pages: Détails par page (pour les PDF)
    
    Raises:
        HTTPException 404: Fichier non trouvé
        HTTPException 500: Erreur lors de la classification
    """
    try:
        # Vérification de l'existence du fichier
        file_path = UPLOAD_DIR / file_id
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Fichier non trouvé")
        
        is_pdf = file_path.suffix.lower() == '.pdf'
        
        # Classification via le pipeline (lazy loading du classifieur)
        classifier_instance = get_classifier_instance()
        results = classifier_instance.classify_document(file_path, is_pdf=is_pdf)
        
        # Ajout des métadonnées
        results["file_id"] = file_id
        results["is_pdf"] = is_pdf
        
        # Conversion pour sérialisation JSON (gestion des types NumPy)
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
    """
    Upload et classification en une seule requête.
    Combine /api/upload et /api/classify pour plus de simplicité.
    
    Args:
        file: Fichier à classifier
    
    Returns:
        Résultats de classification (même format que /api/classify)
    """
    try:
        # Upload du fichier
        upload_result = await upload_file(file)
        file_id = upload_result["file_id"]
        
        # Classification immédiate
        classify_result = await classify_document(file_id)
        
        return classify_result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.get("/api/file/{file_id}")
async def get_file(file_id: str):
    """
    Récupération d'un fichier uploadé.
    Utilisé pour afficher l'image dans la page de résultats.
    
    Args:
        file_id: Identifiant du fichier
    
    Returns:
        Le fichier demandé
    
    Raises:
        HTTPException 404: Fichier non trouvé
    """
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
    """
    Endpoint de debug pour obtenir des informations détaillées.
    Inclut l'état des modèles et le device utilisé (CPU/GPU).
    
    Args:
        file_id: Identifiant du fichier
    
    Returns:
        Résultats de classification + informations de debug
    """
    try:
        file_path = UPLOAD_DIR / file_id
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Fichier non trouvé")
        
        is_pdf = file_path.suffix.lower() == '.pdf'
        results = classifier.classify_document(file_path, is_pdf=is_pdf)
        
        # Ajout des informations de debug
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


# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Démarrage du serveur Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
