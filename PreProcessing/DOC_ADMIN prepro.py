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

dataset_folder = r"C:\NLP-CV\dataset\document administrative"
output_file = r"C:\NLP-CV\dataset\document_admin_dataset.xlsx"

def detect_document_type(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    try:
        text_full = pytesseract.image_to_string(gray, lang="fra+ara", config="--oem 3 --psm 6", timeout=120)
    except:
        try:
            text_full = pytesseract.image_to_string(gray, lang="fra+ara", config="--oem 3 --psm 11", timeout=120)
        except:
            text_full = pytesseract.image_to_string(gray, lang="fra", config="--oem 3 --psm 6", timeout=120)
    
    doc_type = "Non identifiée"
    upper_text = text_full.upper()
    
    keywords_contrat = ['CONTRAT', 'CONTRACT', 'EMPLOI', 'TRAVAIL', 'SALARIE', 'ENGAGEMENT', 'PERIODE D\'ESSAI', 'DUREE DETERMINEE', 'CDI', 'CDD']
    keywords_attestation = ['ATTESTATION', 'CERTIFICATE', 'CERTIFICAT', 'DECLARE', 'CONFIRME', 'ATTESTE']
    keywords_fiche_paie = ['FICHE DE PAIE', 'BULLETIN DE SALAIRE', 'SALAIRE', 'NET A PAYER', 'COTISATIONS', 'CNSS', 'CIMR', 'AMO', 'IR', 'BASE IMPOSABLE']
    keywords_avis = ['AVIS', 'NOTICE', 'DECISION', 'NOTIFICATION']
    keywords_certificat = ['CERTIFICAT DE TRAVAIL', 'CERTIFICAT MEDICAL', 'CERTIFICAT DE RESIDENCE']
    keywords_lettre = ['LETTRE', 'LETTER', 'MOTIF', 'OBJET']
    
    scores = {
        'Contrat': sum(1 for kw in keywords_contrat if kw in upper_text),
        'Attestation': sum(1 for kw in keywords_attestation if kw in upper_text),
        'Fiche de paie': sum(1 for kw in keywords_fiche_paie if kw in upper_text),
        'Avis': sum(1 for kw in keywords_avis if kw in upper_text),
        'Certificat': sum(1 for kw in keywords_certificat if kw in upper_text),
        'Lettre': sum(1 for kw in keywords_lettre if kw in upper_text)
    }
    
    max_score = max(scores.values())
    if max_score > 0:
        doc_type = max(scores, key=scores.get)
    elif 'CONTRAT' in upper_text:
        doc_type = 'Contrat'
    elif 'ATTESTATION' in upper_text:
        doc_type = 'Attestation'
    elif 'FICHE' in upper_text and 'PAIE' in upper_text:
        doc_type = 'Fiche de paie'
    elif 'BULLETIN' in upper_text and 'SALAIRE' in upper_text:
        doc_type = 'Fiche de paie'
    elif 'AVIS' in upper_text:
        doc_type = 'Avis'
    elif 'CERTIFICAT' in upper_text:
        doc_type = 'Certificat'
    elif 'LETTRE' in upper_text:
        doc_type = 'Lettre'
    
    return doc_type, text_full

def process_image(img_array):
    doc_type, text_full = detect_document_type(img_array)
    return doc_type, text_full

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
            doc_type, text_full = process_image(img)
            results.append((doc_type, text_full, page_num + 1))
        pdf_doc.close()
    elif HAS_PDF2IMAGE:
        images = convert_from_path(path)
        for page_num, pil_img in enumerate(images):
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            doc_type, text_full = process_image(img)
            results.append((doc_type, text_full, page_num + 1))
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
                detected_type = None
                for doc_type, text_full, page_num in results:
                    if detected_type is None and doc_type != "Non identifiée":
                        detected_type = doc_type
                    elif detected_type is None:
                        detected_type = doc_type
                
                if detected_type is None:
                    detected_type = "Non identifiée"
                
                for doc_type, text_full, page_num in results:
                    dataset.append({
                        "input": text_full,
                        "output": detected_type,
                        "file": filename,
                        "page": page_num,
                        "type": detected_type
                    })
                    print(f"  Page {page_num}: {detected_type}")
            else:
                pil_img = Image.open(filepath)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                doc_type, text_full = process_image(img)
                dataset.append({
                    "input": text_full,
                    "output": doc_type,
                    "file": filename,
                    "page": 1,
                    "type": doc_type
                })
                print(f"  Type: {doc_type}")
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

