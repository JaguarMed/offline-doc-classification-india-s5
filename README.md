# CV-NLP: Classification de Documents Marocains

Système de classification de documents marocains utilisant 3 modules complémentaires:
1. **OCR + NLP**: Extraction de texte (Tesseract) + Classification Transformer (mDeBERTa fine-tuné)
2. **Vision**: Classification d'images via Vision Transformer (Swin-Tiny/DeiT-Small fine-tuné)
3. **ORB**: Détection de motifs visuels via ORB (Oriented FAST and Rotated BRIEF)

Fusion finale via **soft voting pondéré**.

## 📋 Classes de Documents

- `CIN`: Carte d'Identité Nationale
- `releve_bancaire`: Relevé bancaire
- `facture_eau`: Facture d'eau
- `facture_electricite`: Facture d'électricité
- `document_employeur`: Document employeur (fiche de paie)

## 🚀 Installation

### Prérequis

- Python 3.8+
- Tesseract OCR installé sur le système
  - Windows: Télécharger depuis [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
  - Linux: `sudo apt-get install tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara`
  - macOS: `brew install tesseract tesseract-lang`

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## 📁 Structure du Projet

```
NLP-CV/
├── config/
│   └── config.yaml          # Configuration du pipeline
├── modules/
│   ├── ocr_nlp/             # Module OCR + NLP
│   ├── vision/              # Module Vision
│   ├── orb/                 # Module ORB
│   └── fusion/              # Module de fusion
├── pipeline/
│   └── inference.py         # Pipeline d'inférence unifié
├── tools/
│   ├── build_dataset.py     # Script pour préparer dataset avec OCR
│   └── evaluate.py          # Script pour évaluer un checkpoint
├── app/
│   └── analysis_app.py      # Interface Streamlit d'analyse
├── tests/
│   └── test_smoke.py        # Tests smoke
├── checkpoints/
│   ├── text/                # Checkpoints modèles texte (à placer ici après entraînement Colab)
│   └── vision/              # Checkpoints modèles vision (à placer ici après entraînement Colab)
├── runs/                    # Résultats d'évaluation (générés automatiquement)
├── dataset/                  # Dataset d'entraînement/test
└── cache/                    # Cache OCR
```

## 🔧 Configuration

Éditer `config/config.yaml` pour ajuster:
- Chemins des checkpoints
- Type de modèle vision (`swin_tiny` ou `deit_small`)
- Poids de fusion
- Langues OCR
- Paramètres d'inférence

## 📊 Utilisation

### 1. Préparer le Dataset avec OCR

```bash
python -m tools.build_dataset --input dataset/ --output data_ocr.csv --languages fra+ara
```

Génère un CSV avec les textes OCR extraits pour chaque image.

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

Génère dans `runs/run_001/`:
- `config.json`: Configuration du run
- `metrics.json`: Métriques (accuracy, F1, etc.)
- `predictions.csv`: Prédictions détaillées
- `confusion_matrix.csv` et `.png`: Matrice de confusion
- `errors/`: Top erreurs

### 3. Lancer l'Interface d'Analyse

```bash
streamlit run app/analysis_app.py
```

L'interface propose 3 pages:
- **Run Browser**: Explorer les runs, métriques, confusion matrix
- **Error Explorer**: Analyser les erreurs avec filtres
- **Quick Test**: Tester une image/PDF en temps réel

## 🧪 Tests

```bash
pytest tests/test_smoke.py -v
```

Tests smoke pour vérifier:
- Chargement des checkpoints
- Pipeline d'inférence sur 1 image
- Génération de runs

## 📦 Checkpoints (Colab)

Après entraînement sur Colab, placer les checkpoints dans:

- `checkpoints/text/`: Modèle mDeBERTa fine-tuné (doit contenir `config.json`, `pytorch_model.bin`, `tokenizer_config.json`, etc.)
- `checkpoints/vision/`: Modèle Vision Transformer fine-tuné (fichier `.pth` ou `.pt`)

### Format attendu pour Text Checkpoint

```
checkpoints/text/
├── config.json
├── pytorch_model.bin (ou model.safetensors)
├── tokenizer_config.json
├── vocab.txt
└── ...
```

### Format attendu pour Vision Checkpoint

Fichier `.pth` ou `.pt` contenant le `state_dict` du modèle fine-tuné.

## 🎯 Pipeline d'Inférence

Le pipeline suit cette logique:

1. **OCR**: Extraction de texte depuis l'image/PDF (avec cache)
2. **Text Module**: Classification du texte via mDeBERTa
3. **Vision Module**: Classification de l'image via ViT
4. **ORB Module**: Détection de motifs visuels
5. **Fusion**: Soft voting pondéré des 3 modules
6. **Résultat**: Label final + confidence + détails par module

## 🔍 Optimisations CPU

- `torch.inference_mode()` pour l'inférence
- Cache OCR pour éviter les recalculs
- Redimensionnement d'images optimisé
- Threads configurés pour PyTorch

## 📝 Notes

- L'entraînement se fait sur Colab (Optuna, fine-tuning)
- Cette interface sert uniquement à l'**analyse** et au **test** (pas de tuning)
- Les modèles doivent être fine-tunés en 5 classes avant utilisation

## 🐛 Dépannage

### Erreur Tesseract

Vérifier que Tesseract est installé et dans le PATH:
```bash
tesseract --version
```

### Erreur PDF

Installer `poppler` pour `pdf2image`:
- Windows: Télécharger depuis [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases)
- Linux: `sudo apt-get install poppler-utils`
- macOS: `brew install poppler`

### Erreur Checkpoint

Vérifier que les checkpoints sont bien placés dans `checkpoints/text/` et `checkpoints/vision/`.

## 📄 Licence

Projet interne - Classification de documents marocains.









