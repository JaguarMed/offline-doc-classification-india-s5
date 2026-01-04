"""Script pour préparer le dataset avec OCR."""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Ajouter le root au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.ocr_nlp import extract_ocr_text


def build_dataset(input_dir: str, output_file: str, languages: str = "fra+ara"):
    """
    Construit un dataset avec OCR à partir d'un dossier structuré.
    
    Structure attendue:
    input_dir/
      CIN/
        *.png
      releve_bancaire/
        *.png
      ...
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    # Créer le dossier de sortie si nécessaire
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Parcourir les sous-dossiers (classes)
    data = []
    
    for class_dir in input_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        if class_name.startswith('.'):
            continue
        
        print(f"Traitement de la classe: {class_name}")
        
        # Parcourir les images
        image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.pdf"))
        
        for img_path in tqdm(image_files, desc=f"  {class_name}"):
            try:
                # OCR
                ocr_text = extract_ocr_text(
                    img_path,
                    languages=languages,
                    cache_dir="./cache/ocr",
                    enable_cache=True
                )
                
                data.append({
                    'file_path': str(img_path),
                    'label': class_name,
                    'ocr_text': ocr_text,
                    'ocr_length': len(ocr_text)
                })
            except Exception as e:
                print(f"Erreur sur {img_path}: {e}")
                continue
    
    # Créer DataFrame et sauvegarder
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nDataset créé: {output_path}")
    print(f"Total: {len(df)} fichiers")
    print(f"Par classe:\n{df['label'].value_counts()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construire un dataset avec OCR")
    parser.add_argument("--input", type=str, required=True, help="Dossier d'entrée (structure par classe)")
    parser.add_argument("--output", type=str, required=True, help="Fichier CSV de sortie")
    parser.add_argument("--languages", type=str, default="fra+ara", help="Langues OCR (ex: fra+ara)")
    
    args = parser.parse_args()
    build_dataset(args.input, args.output, args.languages)







