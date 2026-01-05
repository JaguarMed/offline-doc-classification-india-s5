# API de Classification de Documents Marocains

API FastAPI pour la classification de documents marocains utilisant 3 modèles: VLM (Qwen2-VL), ORB, et XLM-R.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Les chemins par défaut sont configurés dans `api/config.py`. Vous pouvez les modifier ou utiliser des variables d'environnement:

- `DATASET_ROOT`: Chemin vers le dataset (défaut: `C:\NLP-CV\dataset`)
- `MODEL_XLMR_PATH`: Chemin vers le modèle XLM-R (défaut: `models_checkpoint/best_model_xlmr_large`)

## Démarrage

### Backend API

```bash
cd api
python main.py
```

L'API sera accessible sur `http://localhost:8000`

### Frontend Web

Ouvrir `web/index.html` dans un navigateur web moderne, ou servir les fichiers statiques:

```bash
# Option 1: Python simple HTTP server
cd web
python -m http.server 8080

# Option 2: Utiliser FastAPI (déjà configuré)
# Les fichiers dans web/ sont servis sur /static
```

## Endpoints API

### `GET /`
Health check basique

### `GET /health`
Status de l'API et disponibilité des modèles

### `POST /api/upload`
Upload un fichier (image ou PDF)

**Request:**
- `file`: Fichier (multipart/form-data)

**Response:**
```json
{
  "filename": "20241231_120000_document.pdf",
  "file_id": "20241231_120000_document.pdf",
  "is_pdf": true,
  "size": 123456
}
```

### `POST /api/classify`
Classifier un document uploadé

**Request:**
- `file_id`: ID du fichier (query parameter)

**Response:**
```json
{
  "file_id": "20241231_120000_document.pdf",
  "is_pdf": true,
  "pages": [
    {
      "page": 1,
      "vlm_level1": {...},
      "vlm_level2": {...},
      "orb": {...},
      "xlmr": {...},
      "final_level1": "RELEVE BANCAIRE",
      "final_level2": "Releve bancaire BANQUE POPULAIRE",
      "final_confidence": 0.95
    }
  ],
  "final_level1": "RELEVE BANCAIRE",
  "final_level2": "Releve bancaire BANQUE POPULAIRE",
  "final_confidence": 0.95,
  "models_results": {
    "vlm_available": true,
    "orb_available": true,
    "xlmr_available": true
  }
}
```

### `POST /api/classify/upload`
Upload et classifier en une seule requête (convenience endpoint)

### `GET /api/file/{file_id}`
Récupérer un fichier uploadé

### `GET /api/debug/{file_id}`
Version debug avec tous les détails (temps, erreurs, etc.)

## Pipeline de Classification

1. **VLM (Qwen2-VL)**: Classification principale Level 1 (4 classes) et Level 2 (12+ classes via SigLIP RAG)
2. **ORB**: Renforcement visuel (logo/template matching) - poids 0.3
3. **XLM-R**: Renforcement textuel (si OCR disponible) - poids 0.2

**Fusion:** `0.5 * VLM + 0.3 * ORB + 0.2 * XLM-R`

## Exemples

### cURL

```bash
# Upload
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@document.pdf"

# Classify
curl -X POST "http://localhost:8000/api/classify?file_id=20241231_120000_document.pdf"

# Upload + Classify
curl -X POST "http://localhost:8000/api/classify/upload" \
  -F "file=@document.pdf"
```

### Python

```python
import requests

# Upload
with open('document.pdf', 'rb') as f:
    response = requests.post('http://localhost:8000/api/upload', files={'file': f})
    file_id = response.json()['file_id']

# Classify
response = requests.post(f'http://localhost:8000/api/classify?file_id={file_id}')
results = response.json()

print(f"Level 1: {results['final_level1']}")
print(f"Level 2: {results['final_level2']}")
print(f"Confidence: {results['final_confidence']:.2%}")
```

## Notes

- Les modèles sont chargés au démarrage de l'API (lazy loading pour SigLIP RAG)
- Support PDF multi-pages: chaque page est traitée indépendamment, puis vote final
- Gestion d'erreurs: si un modèle échoue, les autres continuent
- Fallback CPU automatique si CUDA n'est pas disponible
- Timeout: VLM (30s), ORB (10s), XLM-R (15s) - à configurer dans `api/config.py`







