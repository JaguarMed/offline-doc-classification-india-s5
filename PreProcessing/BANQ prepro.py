import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import io
import pandas as pd
from pathlib import Path
try:
    import fitz
    HAS_FITZ = True
except:
    HAS_FITZ = False
    try:
        from pdf2image import convert_from_path
        HAS_PDF2IMAGE = True
    except:
        HAS_PDF2IMAGE = False

pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract\tesseract.exe"

dataset_folder = r"C:\NLP-CV\dataset\releve bancaire"
output_file = r"C:\NLP-CV\dataset\releve_bancaire_dataset.xlsx"

def detect_bank(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    try:
        text_full = pytesseract.image_to_string(gray, lang="fra+ara", config="--oem 3 --psm 6", timeout=120)
    except:
        try:
            text_full = pytesseract.image_to_string(gray, lang="fra+ara", config="--oem 3 --psm 11", timeout=120)
        except:
            text_full = pytesseract.image_to_string(gray, lang="fra", config="--oem 3 --psm 6", timeout=120)
    
    bank = "Non identifiée"
    keywords = ['CDM', 'SOCIETE GENERALE', 'BANK', 'BANQUE', 'POPULAIRE', 'BARID', 'ATTIJARI', 'CIH', 'CREDIT', 'AGRICOLE', 'MAROC']
    upper_text_full = text_full.upper()
    
    for kw in keywords:
        if kw in upper_text_full:
            lines = text_full.split('\n')
            for line in lines:
                line_clean = line.strip()
                if len(line_clean) < 3:
                    continue
                upper_line = line_clean.upper()
                if kw in upper_line:
                    words = line_clean.split()
                    bank_idx = -1
                    for i, word in enumerate(words):
                        if kw in word.upper():
                            bank_idx = i
                            break
                    if bank_idx >= 0:
                        if kw == 'CDM':
                            bank = 'CDM CREDIT DU MAROC'
                        elif kw == 'SOCIETE GENERALE' or 'SOCIETE' in upper_line and 'GENERALE' in upper_line:
                            bank = 'SOCIETE GENERALE'
                        elif kw == 'CREDIT' and 'MAROC' in upper_line:
                            bank = 'CREDIT DU MAROC'
                        else:
                            start = max(0, bank_idx - 2)
                            end = min(len(words), bank_idx + 3)
                            bank_words = words[start:end]
                            filtered = []
                            for w in bank_words:
                                w_clean = w.strip('.,;:!?()[]{}"\'').strip()
                                w_clean = ''.join(c for c in w_clean if c.isprintable() and ord(c) < 128)
                                if len(w_clean) > 1 and len(w_clean) < 25:
                                    filtered.append(w_clean)
                            if filtered:
                                bank = ' '.join(filtered).strip()
                                bank = ''.join(c for c in bank if c.isprintable() and ord(c) < 128).strip()
                                
                                prefixes_to_remove = ['AN', 'DA', 'NN', 'LM', 'DE', 'LE', 'LA', 'DU', 'DES', 'LES']
                                bank_words_split = bank.split()
                                if len(bank_words_split) > 1 and bank_words_split[0].upper() in prefixes_to_remove:
                                    bank = ' '.join(bank_words_split[1:]).strip()
                                
                                if len(bank) < 3:
                                    bank = line_clean[:60].strip()
                                    bank = ''.join(c for c in bank if c.isprintable() and ord(c) < 128).strip()
                        if bank != "Non identifiée":
                            break
                    else:
                        if kw == 'CDM':
                            bank = 'CDM CREDIT DU MAROC'
                        elif kw == 'SOCIETE GENERALE':
                            bank = 'SOCIETE GENERALE'
                    if bank != "Non identifiée":
                        break
            if bank != "Non identifiée":
                break
    return bank, text_full

def process_image(img_array):
    bank, text_full = detect_bank(img_array)
    return bank, text_full

def process_pdf(path):
    results = []
    if HAS_FITZ:
        pdf_doc = fitz.open(path)
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("ppm")
            pil_img = Image.open(io.BytesIO(img_data))
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            bank, text_full = process_image(img)
            results.append((bank, text_full, page_num + 1))
        pdf_doc.close()
    elif HAS_PDF2IMAGE:
        images = convert_from_path(path)
        for page_num, pil_img in enumerate(images):
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            bank, text_full = process_image(img)
            results.append((bank, text_full, page_num + 1))
    return results

dataset = []

if os.path.isdir(dataset_folder):
    files = [f for f in os.listdir(dataset_folder) if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg'))]
    print(f"Traitement de {len(files)} fichiers...\n")
    
    for filename in files:
        filepath = os.path.join(dataset_folder, filename)
        print(f"Traitement: {filename}")
        
        try:
            if filename.lower().endswith('.pdf'):
                results = process_pdf(filepath)
                detected_bank = None
                for bank, text_full, page_num in results:
                    if detected_bank is None and bank != "Non identifiée":
                        detected_bank = bank
                    elif detected_bank is None:
                        detected_bank = bank
                
                if detected_bank is None:
                    detected_bank = "Non identifiée"
                
                for bank, text_full, page_num in results:
                    output_text = f"Releve bancaire {detected_bank}"
                    dataset.append({
                        "input": text_full,
                        "output": output_text,
                        "file": filename,
                        "page": page_num,
                        "bank": detected_bank
                    })
                    print(f"  Page {page_num}: {detected_bank}")
            else:
                pil_img = Image.open(filepath)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                bank, text_full = process_image(img)
                output_text = f"Releve bancaire {bank}"
                dataset.append({
                    "input": text_full,
                    "output": output_text,
                    "file": filename,
                    "page": 1,
                    "bank": bank
                })
                print(f"  Banque: {bank}")
        except Exception as e:
            print(f"  Erreur: {e}")
            import traceback
            traceback.print_exc()
            continue
        print()

df = pd.DataFrame(dataset)
df.to_excel(output_file, index=False, engine='openpyxl')

print(f"\nDataset cree: {output_file}")
print(f"Nombre d'entrees: {len(dataset)}")
print(f"Colonnes: {list(df.columns)}")
