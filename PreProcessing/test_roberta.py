import torch
import pandas as pd
import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = r"C:\NLP-CV\checkpoints"
dataset_file = r"C:\NLP-CV\dataset\Data.xlsx"

USE_EASYOCR = False
try:
    import pytesseract
    USE_EASYOCR = True
    print("EasyOCR disponible - utilisation d'EasyOCR pour OCR")
except ImportError:
    print("EasyOCR non disponible - utilisation de Tesseract")
    import pytesseract
    from PIL import Image
    import cv2
    import numpy as np
    pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract\tesseract.exe"

if USE_EASYOCR:
    reader = easyocr.Reader(['fr', 'en', 'ar'], gpu=torch.cuda.is_available())

print("Verification GPU...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device utilise: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoire GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("\n" + "="*60 + "\n")

possible_paths = [
    r"C:\NLP-CV\models_checkpoint",
    r"C:\NLP-CV\checkpoints\models",
    r"C:\NLP-CV\checkpoints\text",
    r"C:\NLP-CV\checkpoints",
    r"C:\NLP-CV"
]

model_dirs = []
for base_path in possible_paths:
    if os.path.exists(base_path):
        for root, dirs, files in os.walk(base_path):
            for d in dirs:
                full_path = os.path.join(root, d)
                config_path = os.path.join(full_path, 'config.json')
                if os.path.exists(config_path):
                    if 'roberta' in d.lower() or 'large' in d.lower():
                        model_dirs.append(full_path)
                    else:
                        try:
                            import json
                            with open(config_path, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                                model_type = config.get('model_type', '').lower()
                                arch = config.get('architectures', [])
                                if 'roberta' in model_type or any('roberta' in str(a).lower() for a in arch):
                                    model_dirs.append(full_path)
                        except:
                            pass

if not model_dirs:
    print("Aucun modele RoBERTa trouve automatiquement.")
    print("\nVeuillez specifier le chemin du modele.")
    print("Exemple: C:\\NLP-CV\\checkpoints\\models\\roberta-large")
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
        print(f"Utilisation du chemin fourni: {model_dir}")
    else:
        print("Pour utiliser un chemin specifique, executez:")
        print(f'python "{os.path.basename(__file__)}" "chemin/vers/modele"')
        sys.exit(1)
else:
    model_dir = model_dirs[0]
    print(f"Modele trouve: {model_dir}")

print(f"Chargement du modele depuis: {model_dir}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    print("Modele charge avec succes!")
except Exception as e:
    print(f"Erreur lors du chargement du modele: {e}")
    exit()

print("\n" + "="*60 + "\n")

if os.path.exists(dataset_file):
    df = pd.read_excel(dataset_file)
    print(f"Dataset charge: {len(df)} lignes")
    
    if 'input' in df.columns and 'output' in df.columns:
        cin_df = df[df['output'].str.upper() == 'CIN'].copy()
        
        if len(cin_df) == 0:
            print("Aucune donnee CIN trouvee dans le dataset")
            print(f"Classes disponibles: {df['output'].unique()}")
        else:
            print(f"\nDonnees CIN trouvees: {len(cin_df)} echantillons")
            sample = cin_df.sample(1).iloc[0]
            test_text = str(sample['input'])
            true_label = str(sample.get('output', 'Unknown'))
            
            print(f"\nTest avec un echantillon CIN:")
            print(f"True label: {true_label}")
            try:
                print(f"Texte (premiers 300 caracteres): {test_text[:300]}...")
            except UnicodeEncodeError:
                sys.stdout.buffer.write(f"Texte (premiers 300 caracteres): {test_text[:300].encode('utf-8', errors='ignore')}...\n".encode('utf-8'))
            
            inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class_id = torch.argmax(probabilities, dim=-1).item()
            
            id2label = model.config.id2label if hasattr(model.config, 'id2label') and model.config.id2label else None
            label2id = model.config.label2id if hasattr(model.config, 'label2id') and model.config.label2id else None
            
            if id2label:
                predicted_label = id2label[predicted_class_id]
            else:
                predicted_label = f"Class_{predicted_class_id}"
            
            confidence = probabilities[0][predicted_class_id].item()
            
            print(f"\n{'='*60}")
            print(f"Prediction: {predicted_label}")
            print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            match_result = "OUI" if predicted_label.upper() == true_label.upper() else "NON"
            print(f"Match: {match_result}")
            
            num_classes = probabilities.shape[1]
            k = min(3, num_classes)
            print(f"\nTop {k} predictions:")
            top_probs, top_indices = torch.topk(probabilities[0], k)
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                label = id2label[idx.item()] if id2label else f"Class_{idx.item()}"
                print(f"  {i+1}. {label}: {prob.item():.4f} ({prob.item()*100:.2f}%)")
    else:
        print("Colonnes input/output introuvables dans le dataset")
        print(f"Colonnes disponibles: {list(df.columns)}")
else:
    print(f"Dataset introuvable: {dataset_file}")

print("\n" + "="*60 + "\n")
print("Test termine!")

