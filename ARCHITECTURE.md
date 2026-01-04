# 🏗️ Architecture du Projet CV-NLP

## Vue d'Ensemble

Le projet suit une architecture modulaire avec 3 modules de classification complémentaires + un module de fusion.

```
Image/PDF
    ↓
[OCR] → Texte
    ↓
    ├─→ [Text Classifier] (mDeBERTa) ──┐
    ├─→ [Vision Classifier] (ViT) ──────┤
    └─→ [ORB Classifier] (motifs) ──────┤
                                        ↓
                                  [Fusion Module]
                                        ↓
                                  Prédiction Finale
```

## 📁 Structure des Modules

### 1. Module OCR+NLP (`modules/ocr_nlp/`)

**Fichiers**:
- `ocr.py`: Extraction OCR avec cache
- `text_classifier.py`: Classifieur Transformer texte

**Fonctionnalités**:
- Extraction OCR multi-langue (FR/AR)
- Cache des résultats OCR (hash MD5)
- Pré-traitement image (grayscale, threshold)
- Classification via mDeBERTa fine-tuné
- Nettoyage texte robuste

**API**:
```python
from modules.ocr_nlp import extract_ocr_text, TextClassifier

# OCR
text = extract_ocr_text("image.png", languages="fra+ara")

# Classification
classifier = TextClassifier("checkpoints/text")
result = classifier.predict(text)
# → {'label': 'CIN', 'confidence': 0.95, 'probabilities': {...}}
```

### 2. Module Vision (`modules/vision/`)

**Fichiers**:
- `vision_classifier.py`: Classifieur Vision Transformer

**Fonctionnalités**:
- Support Swin-Tiny et DeiT-Small
- Optimisé pour CPU (inference_mode)
- Redimensionnement automatique (224x224)
- Normalisation standard ImageNet

**API**:
```python
from modules.vision import VisionClassifier

classifier = VisionClassifier(
    "checkpoints/vision",
    model_type="swin_tiny"
)
result = classifier.predict("image.png")
# → {'label': 'CIN', 'confidence': 0.92, 'probabilities': {...}}
```

### 3. Module ORB (`modules/orb/`)

**Fichiers**:
- `orb_classifier.py`: Classifieur basé sur ORB

**Fonctionnalités**:
- Détection de keypoints ORB
- Scores heuristiques basés sur:
  - Densité de features
  - Ratio d'aspect
  - Détection de lignes (tableaux)
- Probabilités normalisées par classe

**API**:
```python
from modules.orb import ORBClassifier

classifier = ORBClassifier()
result = classifier.predict("image.png")
# → {'label': 'CIN', 'confidence': 0.65, 'probabilities': {...}, 'metadata': {...}}
```

### 4. Module Fusion (`modules/fusion/`)

**Fichiers**:
- `fusion.py`: Soft voting pondéré

**Fonctionnalités**:
- Fusion pondérée des probabilités
- Poids configurables (text, vision, orb)
- Normalisation automatique
- Détails par module dans le résultat

**API**:
```python
from modules.fusion import FusionModule

fusion = FusionModule(weight_text=0.6, weight_vision=0.3, weight_orb=0.1)
result = fusion.fuse(text_pred, vision_pred, orb_pred)
# → {'label': 'CIN', 'confidence': 0.88, 'probabilities': {...}, 'module_details': {...}}
```

## 🔄 Pipeline d'Inférence (`pipeline/inference.py`)

**Classe**: `InferencePipeline`

**Responsabilités**:
- Orchestration des 3 modules
- Gestion des erreurs (fallback sur prédictions neutres)
- Configuration via YAML ou paramètres directs
- Support CPU/CUDA automatique

**Flux d'exécution**:
1. Charger image/PDF
2. Extraire OCR (avec cache)
3. Prédire via Text Classifier
4. Prédire via Vision Classifier
5. Prédire via ORB Classifier
6. Fusionner les 3 prédictions
7. Retourner résultat complet

**API**:
```python
from pipeline.inference import InferencePipeline

pipeline = InferencePipeline(
    config_path="config/config.yaml",
    text_checkpoint="checkpoints/text",
    vision_checkpoint="checkpoints/vision"
)

result = pipeline.predict("document.png", return_ocr_text=True, return_details=True)
# → {
#     'prediction': {'label': 'CIN', 'confidence': 0.88, ...},
#     'ocr_text': '...',
#     'module_predictions': {...}
# }
```

## 🛠️ Outils CLI (`tools/`)

### `build_dataset.py`

**Rôle**: Préparer dataset avec OCR

**Entrée**: Dossier structuré par classe
```
dataset/
├── CIN/
│   ├── CIN_001.png
│   └── ...
├── releve_bancaire/
│   └── ...
```

**Sortie**: CSV avec colonnes:
- `file_path`
- `label`
- `ocr_text`
- `ocr_length`

### `evaluate.py`

**Rôle**: Évaluer un checkpoint et générer artefacts

**Entrée**:
- Dataset (structure par classe)
- Checkpoints texte et vision

**Sortie**: Dossier `runs/<run_id>/` avec:
- `config.json`: Configuration du run
- `metrics.json`: Métriques globales et par classe
- `predictions.csv`: Prédictions détaillées
- `confusion_matrix.csv` + `.png`
- `errors/top_errors.csv`

## 🖥️ Interface Streamlit (`app/analysis_app.py`)

**Pages**:

### 1. Run Browser
- Sélection d'un run
- KPIs (accuracy, F1, etc.)
- Confusion matrix interactive
- Table métriques par classe
- Distribution des confidences
- Top erreurs

### 2. Error Explorer
- Filtres: true_label, pred_label, low confidence, short OCR
- Liste d'erreurs
- Détails par erreur:
  - Image
  - Prédiction finale + breakdown
  - Texte OCR

### 3. Quick Test
- Upload image/PDF
- Prédiction en temps réel
- Breakdown par module
- Probabilités complètes
- Texte OCR

## 📊 Format des Données

### Prédiction d'un Module

```python
{
    'label': 'CIN',
    'confidence': 0.95,
    'probabilities': {
        'CIN': 0.95,
        'releve_bancaire': 0.02,
        'facture_eau': 0.01,
        'facture_electricite': 0.01,
        'document_employeur': 0.01
    }
}
```

### Résultat du Pipeline

```python
{
    'image_path': 'path/to/image.png',
    'prediction': {
        'label': 'CIN',
        'confidence': 0.88,
        'probabilities': {...},
        'module_details': {
            'text': {'label': 'CIN', 'confidence': 0.95},
            'vision': {'label': 'CIN', 'confidence': 0.92},
            'orb': {'label': 'CIN', 'confidence': 0.65}
        }
    },
    'ocr_text': '...',
    'module_predictions': {
        'text': {...},
        'vision': {...},
        'orb': {...}
    }
}
```

## 🔧 Configuration (`config/config.yaml`)

**Sections**:
- `classes`: Liste des 5 classes
- `checkpoints`: Chemins des checkpoints
- `vision_model`: Type de modèle vision
- `fusion_weights`: Poids de fusion
- `ocr`: Configuration OCR (langues, cache)
- `orb`: Paramètres ORB
- `inference`: Paramètres d'inférence (batch_size, threads, etc.)

## 🧪 Tests (`tests/test_smoke.py`)

**Tests smoke**:
1. `test_ocr_extraction`: Extraction OCR sur une image
2. `test_text_classifier_loading`: Chargement classifieur texte
3. `test_vision_classifier_loading`: Chargement classifieur vision
4. `test_orb_classifier`: Classifieur ORB
5. `test_fusion_module`: Module de fusion
6. `test_inference_pipeline`: Pipeline complet

**Note**: Les tests skip automatiquement si les checkpoints ne sont pas disponibles.

## 🚀 Optimisations CPU

1. **Cache OCR**: Hash MD5 pour éviter recalculs
2. **torch.inference_mode()**: Mode inférence optimisé
3. **Redimensionnement**: Images redimensionnées avant traitement
4. **Threads**: Configuration PyTorch pour CPU multi-thread
5. **Lazy Loading**: Modules chargés uniquement si checkpoints disponibles

## 📦 Dépendances Principales

- **PyTorch**: Modèles deep learning
- **Transformers**: mDeBERTa
- **Tesseract**: OCR
- **OpenCV**: Traitement images, ORB
- **Streamlit**: Interface web
- **Plotly**: Visualisations interactives
- **scikit-learn**: Métriques d'évaluation

## 🔄 Workflow Typique

1. **Préparation** (local):
   ```bash
   python -m tools.build_dataset --input dataset/ --output data_ocr.csv
   ```

2. **Entraînement** (Colab):
   - Fine-tuning mDeBERTa sur textes OCR
   - Fine-tuning ViT sur images
   - Télécharger checkpoints

3. **Évaluation** (local):
   ```bash
   python -m tools.evaluate --dataset dataset/ --text_ckpt ... --vision_ckpt ... --out runs/run_001
   ```

4. **Analyse** (local):
   ```bash
   streamlit run app/analysis_app.py
   ```

## 🎯 Points d'Extension

- **Nouveaux modules**: Ajouter dans `modules/` et intégrer dans `InferencePipeline`
- **Nouvelles métriques**: Étendre `tools/evaluate.py`
- **Nouvelles visualisations**: Ajouter dans `app/analysis_app.py`
- **Support batch**: Ajouter méthode `predict_batch()` dans le pipeline
- **API REST**: Créer wrapper Flask/FastAPI autour du pipeline










