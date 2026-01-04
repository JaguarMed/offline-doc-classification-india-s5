"""
Script pour créer un mapping des chemins d'images vers les classes exactes (output_precision)
depuis les datasets Excel.
Ce mapping sera utilisé par test_rag.py pour identifier les classes exactes au lieu des noms de dossiers.
"""

import pandas as pd
import os
import json
from pathlib import Path

DATASET_ROOT = r"C:\NLP-CV\dataset"
OUTPUT_MAPPING_FILE = "image_to_label_mapping.json"

def build_label_mapping():
    """Construit un mapping chemin_image -> classe_exacte depuis Data.xlsx"""
    
    data_file = os.path.join(DATASET_ROOT, "Data.xlsx")
    if not os.path.exists(data_file):
        print(f"Fichier {data_file} introuvable")
        return None
    
    df = pd.read_excel(data_file)
    print(f"Dataset chargé: {len(df)} lignes")
    print(f"Colonnes: {list(df.columns)}")
    
    if 'output_precision' not in df.columns:
        print("Colonne 'output_precision' introuvable")
        return None
    
    # Pour les datasets individuels, nous devons créer un mapping depuis les fichiers sources
    mapping = {}
    
    # 1. Relevés bancaires
    banq_file = os.path.join(DATASET_ROOT, "releve_bancaire_dataset.xlsx")
    if os.path.exists(banq_file):
        df_banq = pd.read_excel(banq_file)
        if 'file' in df_banq.columns and 'output' in df_banq.columns:
            for _, row in df_banq.iterrows():
                filename = str(row['file'])
                label = str(row['output'])
                # Créer le chemin relatif depuis le dataset root
                rel_path = f"releve bancaire\\{filename}"
                mapping[rel_path] = label
                # Ajouter aussi avec / pour compatibilité
                mapping[rel_path.replace('\\', '/')] = label
    
    # 2. Factures
    facture_file = os.path.join(DATASET_ROOT, "facture_dataset.xlsx")
    if os.path.exists(facture_file):
        df_facture = pd.read_excel(facture_file)
        if 'file' in df_facture.columns and 'output' in df_facture.columns:
            for _, row in df_facture.iterrows():
                filename = str(row['file'])
                label = str(row['output'])
                rel_path = f"Facture d'eau_d'éléctricité\\{filename}"
                mapping[rel_path] = label
                mapping[rel_path.replace('\\', '/')] = label
    
    # 3. Documents administratifs
    admin_file = os.path.join(DATASET_ROOT, "document_admin_dataset.xlsx")
    if os.path.exists(admin_file):
        df_admin = pd.read_excel(admin_file)
        if 'file' in df_admin.columns and 'output' in df_admin.columns:
            for _, row in df_admin.iterrows():
                filename = str(row['file'])
                label = str(row['output'])
                rel_path = f"document administrative\\{filename}"
                mapping[rel_path] = label
                mapping[rel_path.replace('\\', '/')] = label
    
    # 4. CIN - utiliser le dataset cin_front_back_2cols.xlsx pour avoir les classes exactes
    cin_file = os.path.join(DATASET_ROOT, "cin_front_back_2cols.xlsx")
    if os.path.exists(cin_file):
        df_cin = pd.read_excel(cin_file)
        if 'input' in df_cin.columns and 'output' in df_cin.columns:
            # Le dataset CIN a les chemins dans input (OCR text) - on doit mapper par nom de fichier
            # On va créer un mapping basé sur les noms de fichiers si disponibles
            cin_folder = os.path.join(DATASET_ROOT, "CIN")
            if os.path.exists(cin_folder):
                # Par défaut, on utilise "CIN" mais on devrait chercher dans Data.xlsx pour les classes exactes
                pass
    
    # Utiliser Data.xlsx pour obtenir les classes exactes CIN
    if os.path.exists(data_file):
        df_data = pd.read_excel(data_file)
        if 'output_precision' in df_data.columns:
            # Pour CIN, on ne peut pas mapper directement depuis les fichiers images
            # Mais on peut utiliser "CIN" comme classe par défaut
            cin_folder = os.path.join(DATASET_ROOT, "CIN")
            if os.path.exists(cin_folder):
                for filename in os.listdir(cin_folder):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        rel_path = f"CIN\\{filename}"
                        # Utiliser "CIN" par défaut (les classes exactes CIN_front/CIN_back sont dans Data.xlsx mais sans mapping fichier->classe)
                        mapping[rel_path] = "CIN"
                        mapping[rel_path.replace('\\', '/')] = "CIN"
    
    print(f"\nMapping créé: {len(mapping)} chemins")
    print(f"\nClasses uniques:")
    unique_labels = set(mapping.values())
    for label in sorted(unique_labels):
        count = sum(1 for v in mapping.values() if v == label)
        print(f"  {label}: {count} images")
    
    # Sauvegarder le mapping
    with open(OUTPUT_MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    print(f"\nMapping sauvegardé: {OUTPUT_MAPPING_FILE}")
    return mapping

if __name__ == "__main__":
    build_label_mapping()

