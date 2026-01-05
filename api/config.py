# -*- coding: utf-8 -*-
"""
=============================================================================
CONFIGURATION GLOBALE DE L'APPLICATION
=============================================================================
Ce fichier contient toutes les configurations de l'application :
- Chemins des dossiers et fichiers
- Modèles d'IA utilisés
- Classes de documents supportées
- Timeouts pour les différents modèles

Auteur: Système de Classification de Documents Marocains
=============================================================================
"""

import os
from pathlib import Path

# =============================================================================
# CHEMINS DES DOSSIERS
# =============================================================================

# Dossier racine du projet (parent du dossier api/)
BASE_DIR = Path(__file__).parent.parent

# Chemin vers le dataset d'images (peut être défini via variable d'environnement)
DATASET_ROOT = os.getenv("DATASET_ROOT", r"C:\NLP-CV\dataset")

# Dossier pour stocker les fichiers uploadés par les utilisateurs
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)  # Crée le dossier s'il n'existe pas

# =============================================================================
# MODÈLES D'INTELLIGENCE ARTIFICIELLE
# =============================================================================

# Qwen2-VL-2B : Modèle Vision-Language pour la classification visuelle
# - Taille: 2B paramètres (compatible avec 8GB VRAM)
# - Utilisé pour: Classification Level 1 et détection CIN front/back
MODEL_QWEN = "Qwen/Qwen2-VL-2B-Instruct"

# SigLIP : Modèle d'embeddings visuels pour la recherche par similarité (RAG)
# - Utilisé pour: Classification Level 2 via RAG (Retrieval Augmented Generation)
MODEL_SIGLIP = "google/siglip-base-patch16-224"

# XLM-RoBERTa Large : Modèle de classification de texte multilingue
# - Entraîné sur le texte OCR des documents marocains
# - Supporte: Français, Arabe, Anglais
MODEL_ROBERTA = os.getenv("MODEL_ROBERTA_PATH", str(BASE_DIR / "models_checkpoint" / "best_model_xlmr_large"))

# =============================================================================
# FICHIERS DE CACHE
# =============================================================================

# Cache des embeddings SigLIP (pour RAG Level 2)
EMBEDDINGS_CACHE = BASE_DIR / "embeddings.npy"

# Cache des chemins d'images correspondant aux embeddings
PATHS_CACHE = BASE_DIR / "paths.json"

# Mapping des images vers leurs labels exacts
LABEL_MAPPING_FILE = BASE_DIR / "image_to_label_mapping.json"

# Dossier contenant la galerie ORB pré-calculée
ORB_CACHE_DIR = BASE_DIR / "orb_cache"

# =============================================================================
# CLASSES DE DOCUMENTS SUPPORTÉES
# =============================================================================

# Level 1 : 4 grandes catégories de documents marocains
LEVEL1_CLASSES = [
    "CIN",                              # Carte Nationale d'Identité
    "DOCUMENT EMPLOYEUR",               # Documents d'emploi (attestations, contrats, fiches de paie)
    "FACTURE D'EAU ET D'ELECTRICITE",   # Factures des fournisseurs d'énergie
    "RELEVE BANCAIRE"                   # Relevés de compte bancaire
]

# Level 2 : 14 sous-catégories détaillées
LEVEL2_CLASSES = [
    # CIN (2 variantes)
    "CIN_front",                        # Recto de la CIN (avec photo)
    "CIN_back",                         # Verso de la CIN (avec MRZ/code-barres)
    
    # Relevés bancaires (8 banques marocaines)
    "Releve bancaire Attijariwafa bank",
    "Releve bancaire BANK OF AFRICA",
    "Releve bancaire AL BARID BANK",
    "Releve bancaire BANQUE POPULAIRE",
    "Releve bancaire CREDIT AGRICOLE",
    "Releve bancaire CDM CREDIT DU MAROC",
    "Releve bancaire CIH BANK",
    "Releve bancaire SOCIETE GENERALE",
    
    # Factures (1 catégorie)
    "Facture d'eau et d'électricité",
    
    # Documents employeur (3 types)
    "Attestation",                      # Attestation de travail/salaire
    "Contrat",                          # Contrat de travail
    "Fiche de paie"                     # Bulletin de salaire
]

# =============================================================================
# TIMEOUTS (en secondes)
# =============================================================================

VLM_TIMEOUT = 30      # Timeout pour le modèle VLM Qwen2-VL
ORB_TIMEOUT = 10      # Timeout pour le matching ORB
ROBERTA_TIMEOUT = 15  # Timeout pour la classification RoBERTa
