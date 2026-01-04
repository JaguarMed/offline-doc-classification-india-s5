# 📋 Commandes d'Exécution

## 🚀 Commandes Principales

### 1. Préparer le Dataset avec OCR

```bash
python -m tools.build_dataset --input dataset/ --output data_ocr.csv --languages fra+ara
```

**Description**: Extrait le texte OCR de toutes les images du dataset et génère un CSV.

**Options**:
- `--input`: Dossier contenant les sous-dossiers par classe (CIN/, releve_bancaire/, etc.)
- `--output`: Fichier CSV de sortie
- `--languages`: Langues OCR (défaut: "fra+ara")

**Exemple de sortie**:
```
Traitement de la classe: CIN
  CIN: 100%|████████████| 5/5 [00:02<00:00,  2.1it/s]
...

Dataset créé: data_ocr.csv
Total: 25 fichiers
Par classe:
CIN                     5
releve_bancaire         5
...
```

---

### 2. Évaluer un Checkpoint

```bash
python -m tools.evaluate \
    --dataset dataset/ \
    --text_ckpt checkpoints/text \
    --vision_ckpt checkpoints/vision \
    --out runs/run_001 \
    --config config/config.yaml \
    --vision_model swin_tiny
```

**Description**: Évalue un checkpoint sur le dataset et génère les artefacts d'analyse.

**Options**:
- `--dataset`: Dossier dataset (structure par classe)
- `--text_ckpt`: Chemin vers checkpoint texte (mDeBERTa fine-tuné)
- `--vision_ckpt`: Chemin vers checkpoint vision (ViT fine-tuné)
- `--out`: Dossier de sortie (ex: `runs/run_001`)
- `--config`: Chemin vers config.yaml (optionnel)
- `--vision_model`: Type de modèle vision (`swin_tiny` ou `deit_small`)

**Structure de sortie** (`runs/run_001/`):
```
runs/run_001/
├── config.json              # Configuration du run
├── metrics.json             # Métriques (accuracy, F1, etc.)
├── predictions.csv          # Prédictions détaillées
├── confusion_matrix.csv     # Matrice de confusion (CSV)
├── confusion_matrix.png     # Matrice de confusion (image)
└── errors/
    └── top_errors.csv       # Top 20 erreurs
```

**Exemple de sortie**:
```
Évaluation classe: CIN
  CIN: 100%|████████████| 5/5 [00:15<00:00,  3.2s/it]
...

Évaluation terminée!
Accuracy: 0.9200
Macro F1: 0.9156
Weighted F1: 0.9201

Résultats sauvegardés dans: runs/run_001
```

---

### 3. Lancer l'Interface d'Analyse (Streamlit)

```bash
streamlit run app/analysis_app.py
```

**Description**: Lance l'interface web d'analyse et de test.

**Pages disponibles**:
1. **Run Browser**: Explorer les runs, métriques, confusion matrix
2. **Error Explorer**: Analyser les erreurs avec filtres avancés
3. **Quick Test**: Tester une image/PDF en temps réel

**Accès**: Ouvrir `http://localhost:8501` dans le navigateur

---

## 🧪 Tests

### Lancer les Tests Smoke

```bash
pytest tests/test_smoke.py -v
```

**Tests effectués**:
- Extraction OCR
- Chargement classifieur texte
- Chargement classifieur vision
- Classifieur ORB
- Module de fusion
- Pipeline d'inférence complet

**Note**: Les tests skip automatiquement si les checkpoints ne sont pas disponibles.

---

## 📦 Placement des Checkpoints (après entraînement Colab)

### Checkpoint Texte (mDeBERTa)

Placer dans `checkpoints/text/`:

```
checkpoints/text/
├── config.json
├── pytorch_model.bin          # ou model.safetensors
├── tokenizer_config.json
├── vocab.txt
├── special_tokens_map.json
└── ...
```

**Format**: Modèle HuggingFace Transformers standard (AutoModelForSequenceClassification)

### Checkpoint Vision (ViT)

Placer dans `checkpoints/vision/`:

```
checkpoints/vision/
└── model.pth                  # ou model.pt
```

**Format**: Fichier PyTorch avec `state_dict` du modèle fine-tuné.

**Structure attendue du checkpoint**:
```python
{
    'model_state_dict': {...},  # ou 'state_dict': {...}
    # ou directement le state_dict
}
```

---

## 🔧 Configuration

### Éditer `config/config.yaml`

```yaml
# Chemins des checkpoints
checkpoints:
  text: "./checkpoints/text"
  vision: "./checkpoints/vision"

# Type de modèle vision
vision_model: "swin_tiny"  # ou "deit_small"

# Poids de fusion
fusion_weights:
  text: 0.6
  vision: 0.3
  orb: 0.1

# Langues OCR
ocr:
  languages: "fra+ara"
  cache_dir: "./cache/ocr"
```

---

## 🐛 Dépannage

### Erreur: "Tesseract not found"

**Solution**: Installer Tesseract OCR
- Windows: [Télécharger](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt-get install tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara`
- macOS: `brew install tesseract tesseract-lang`

### Erreur: "poppler not found" (pour PDF)

**Solution**: Installer Poppler
- Windows: [Télécharger](https://github.com/oschwartz10612/poppler-windows/releases)
- Linux: `sudo apt-get install poppler-utils`
- macOS: `brew install poppler`

### Erreur: "Checkpoint not found"

**Vérifier**:
1. Les checkpoints sont bien dans `checkpoints/text/` et `checkpoints/vision/`
2. Les chemins dans `config/config.yaml` sont corrects
3. Les fichiers checkpoint sont complets (pas de corruption)

### Erreur: "CUDA out of memory"

**Solution**: Le pipeline utilise automatiquement CPU si CUDA n'est pas disponible. Pour forcer CPU:
```python
pipeline = InferencePipeline(..., device='cpu')
```

---

## 📊 Exemple de Workflow Complet

```bash
# 1. Préparer dataset avec OCR
python -m tools.build_dataset --input dataset/ --output data_ocr.csv

# 2. (Sur Colab) Entraîner les modèles et télécharger les checkpoints
#    → Placer dans checkpoints/text/ et checkpoints/vision/

# 3. Évaluer le checkpoint
python -m tools.evaluate \
    --dataset dataset/ \
    --text_ckpt checkpoints/text \
    --vision_ckpt checkpoints/vision \
    --out runs/run_001

# 4. Lancer l'interface d'analyse
streamlit run app/analysis_app.py

# 5. Dans l'interface:
#    - Run Browser → Sélectionner "run_001"
#    - Explorer métriques, confusion matrix, erreurs
#    - Quick Test → Tester une nouvelle image
```

---

## 💡 Astuces

- **Cache OCR**: Les résultats OCR sont mis en cache dans `cache/ocr/` pour éviter les recalculs
- **Optimisation CPU**: Le pipeline utilise `torch.inference_mode()` pour optimiser l'inférence
- **Threads**: Configurer `num_threads` dans `config.yaml` pour optimiser PyTorch sur CPU
- **Batch Processing**: Pour traiter plusieurs images, utiliser le pipeline dans une boucle (pas de batch processing intégré pour l'instant)







