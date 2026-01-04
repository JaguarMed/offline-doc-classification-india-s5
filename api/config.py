# -*- coding: utf-8 -*-
"""Configuration pour l'API"""
import os
from pathlib import Path

# Chemins
BASE_DIR = Path(__file__).parent.parent
DATASET_ROOT = os.getenv("DATASET_ROOT", r"C:\NLP-CV\dataset")
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Modèles
# Qwen2-VL-2B - modèle léger compatible avec 8GB VRAM
MODEL_QWEN = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_SIGLIP = "google/siglip-base-patch16-224"
MODEL_ROBERTA = os.getenv("MODEL_ROBERTA_PATH", str(BASE_DIR / "models_checkpoint" / "best_model_xlmr_large"))

# Cache
EMBEDDINGS_CACHE = BASE_DIR / "embeddings.npy"
PATHS_CACHE = BASE_DIR / "paths.json"
LABEL_MAPPING_FILE = BASE_DIR / "image_to_label_mapping.json"
ORB_CACHE_DIR = BASE_DIR / "orb_cache"

# Classes
LEVEL1_CLASSES = [
    "CIN",
    "DOCUMENT EMPLOYEUR",
    "FACTURE D'EAU ET D'ELECTRICITE",
    "RELEVE BANCAIRE"
]

LEVEL2_CLASSES = [
    "CIN_front",
    "CIN_back",
    "Releve bancaire Attijariwafa bank",
    "Releve bancaire BANK OF AFRICA",
    "Releve bancaire AL BARID BANK",
    "Releve bancaire BANQUE POPULAIRE",
    "Releve bancaire CREDIT AGRICOLE",
    "Releve bancaire CDM CREDIT DU MAROC",
    "Releve bancaire CIH BANK",
    "Releve bancaire SOCIETE GENERALE",
    "Facture d'eau et d'électricité",
    "Attestation",
    "Contrat",
    "Fiche de paie"
]

# Timeouts
VLM_TIMEOUT = 30
ORB_TIMEOUT = 10
ROBERTA_TIMEOUT = 15

