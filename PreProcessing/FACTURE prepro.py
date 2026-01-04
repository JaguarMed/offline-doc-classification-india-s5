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

dataset_folder = r"C:\NLP-CV\dataset\Facture d'eau_d'éléctricité"
output_file = r"C:\NLP-CV\dataset\facture_dataset.xlsx"

def detect_facture_type(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    try:
        text_full = pytesseract.image_to_string(gray, lang="fra+ara", config="--oem 3 --psm 6", timeout=120)
    except:
        try:
            text_full = pytesseract.image_to_string(gray, lang="fra+ara", config="--oem 3 --psm 11", timeout=120)
        except:
            text_full = pytesseract.image_to_string(gray, lang="fra", config="--oem 3 --psm 6", timeout=120)
    
    facture_type = "Non identifiée"
    upper_text = text_full.upper()
    
    keywords_facture = ['EAU', 'WATER', 'ELECTRICITE', 'ELECTRICITY', 'ONEE', 'LYDEC', 'RADEEF', 'AMENDIS', 'REDAL', 'REGIE', 'EAUX', 'ENERGIE', 'ENERGY', 'KWH', 'KW', 'FACTURE', 'BILL']
    
    if any(kw in upper_text for kw in keywords_facture):
        facture_type = "Facture d'eau et d'électricité"
    
    return facture_type, text_full

def process_image(img_array):
    facture_type, text_full = detect_facture_type(img_array)
    return facture_type, text_full

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
            facture_type, text_full = process_image(img)
            results.append((facture_type, text_full, page_num + 1))
        pdf_doc.close()
    elif HAS_PDF2IMAGE:
        images = convert_from_path(path)
        for page_num, pil_img in enumerate(images):
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            facture_type, text_full = process_image(img)
            results.append((facture_type, text_full, page_num + 1))
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
                for facture_type, text_full, page_num in results:
                    dataset.append({
                        "input": text_full,
                        "output": facture_type,
                        "file": filename,
                        "page": page_num,
                        "type": facture_type
                    })
                    print(f"  Page {page_num}: {facture_type}")
            else:
                pil_img = Image.open(filepath)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                facture_type, text_full = process_image(img)
                dataset.append({
                    "input": text_full,
                    "output": facture_type,
                    "file": filename,
                    "page": 1,
                    "type": facture_type
                })
                print(f"  Type: {facture_type}")
        except Exception as e:
            print(f"  Erreur: {e}")
            import traceback
            traceback.print_exc()
            continue
        print()

if len(dataset) > 0:
    df = pd.DataFrame(dataset)
    df.to_excel(output_file, index=False, engine='openpyxl')
    
    print(f"\nDataset cree: {output_file}")
    print(f"Nombre d'entrees: {len(dataset)}")
    print(f"Colonnes: {list(df.columns)}")
else:
    print("\nAucune donnee a sauvegarder.")

