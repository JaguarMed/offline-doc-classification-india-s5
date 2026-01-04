# Quick Start - API de Classification

## Installation rapide

```bash
pip install fastapi uvicorn python-multipart
```

## Démarrage

### Option 1: Script batch (Windows)
```bash
start_api.bat
```

### Option 2: Ligne de commande
```bash
cd api
python main.py
```

L'API sera accessible sur `http://localhost:8000`

## Utilisation

### 1. Ouvrir le frontend web

Ouvrir `web/index.html` dans votre navigateur (ou servir avec un serveur HTTP simple).

**Note**: Le frontend doit pointer vers `http://localhost:8000/api` (configuré dans `web/index.html`).

### 2. Tester avec cURL

```bash
# Health check
curl http://localhost:8000/health

# Upload + Classify
curl -X POST "http://localhost:8000/api/classify/upload" -F "file=@votre_document.pdf"
```

### 3. Tester avec Python

```python
import requests

# Upload et classifier
with open('document.pdf', 'rb') as f:
    response = requests.post('http://localhost:8000/api/classify/upload', files={'file': f})
    results = response.json()
    print(f"Level 1: {results['final_level1']}")
    print(f"Level 2: {results['final_level2']}")
```

## Configuration

Modifier `api/config.py` pour changer les chemins:
- `DATASET_ROOT`: Chemin du dataset
- `MODEL_XLMR`: Chemin du modèle XLM-R
- `UPLOAD_DIR`: Dossier pour les uploads (créé automatiquement)

## Troubleshooting

### Erreur "Module not found"
Installer les dépendances:
```bash
pip install -r requirements.txt
```

### Erreur CUDA
L'API bascule automatiquement sur CPU si CUDA n'est pas disponible.

### Erreur "ORB gallery not found"
Reconstruire la galerie ORB:
```bash
python PreProcessing/orb_matcher.py --build_gallery --dataset_root "C:\NLP-CV\dataset"
```

### Erreur "SigLIP index not found"
L'index SigLIP est optionnel. Si absent, seul VLM Level1 sera utilisé (pas de Level2 RAG).





