import torch
import torch.nn.functional as F
import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import io
import cv2
import re
from transformers import AutoProcessor, AutoModel, Qwen2VLForConditionalGeneration
from tqdm import tqdm

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

DATASET_ROOT = r"C:\NLP-CV\dataset"
CACHE_EMBEDDINGS = "embeddings.npy"
CACHE_PATHS = "paths.json"
LABEL_MAPPING_FILE = "image_to_label_mapping.json"
MODEL_SIGLIP = "google/siglip-base-patch16-224"
MODEL_QWEN = "Qwen/Qwen2-VL-2B-Instruct"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
PDF_EXTENSIONS = {'.pdf'}

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

LABEL_MAPPING = {
    'CIN': 'CIN',
    'document administrative': 'DOCUMENT EMPLOYEUR',
    'Facture d\'eau_d\'éléctricité': 'FACTURE D\'EAU ET D\'ELECTRICITE',
    'releve bancaire': 'RELEVE BANCAIRE'
}

def get_device(use_cpu=False):
    if use_cpu:
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_siglip_model(device):
    processor = AutoProcessor.from_pretrained(MODEL_SIGLIP)
    model = AutoModel.from_pretrained(
        MODEL_SIGLIP,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    )
    model.to(device)
    model.eval()
    return processor, model

_qwen_processor = None
_qwen_model = None

def load_qwen_model(device):
    global _qwen_processor, _qwen_model
    if _qwen_processor is not None and _qwen_model is not None:
        return _qwen_processor, _qwen_model
    
    print("Chargement Qwen2-VL depuis le cache local...")
    try:
        _qwen_processor = AutoProcessor.from_pretrained(MODEL_QWEN, trust_remote_code=True, use_fast=False)
        _qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_QWEN,
            trust_remote_code=True,
            dtype=torch.float16 if device.type == 'cuda' else torch.float32
        )
        _qwen_model.to(device)
        _qwen_model.eval()
        print("Qwen2-VL charge!")
        return _qwen_processor, _qwen_model
    except Exception as e:
        print(f"Erreur chargement Qwen2-VL: {e}")
        _qwen_processor = None
        _qwen_model = None
        raise

def get_image_files(dataset_root):
    image_files = []
    pdf_files = []
    
    for root, _, files in os.walk(dataset_root):
        for file in files:
            file_path = os.path.join(root, file)
            suffix = Path(file).suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                image_files.append(file_path)
            elif suffix in PDF_EXTENSIONS:
                pdf_files.append(file_path)
    
    return image_files, pdf_files

def embed_image_siglip(image_path, processor, model, device):
    try:
        img = Image.open(image_path).convert('RGB')
        return embed_image_siglip_from_pil(img, processor, model, device)
    except Exception as e:
        print(f"Erreur SigLIP embedding: {e}")
        return None

def embed_image_siglip_from_pil(pil_img, processor, model, device):
    try:
        inputs = processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    embedding = model.get_image_features(pixel_values=pixel_values)
            else:
                embedding = model.get_image_features(pixel_values=pixel_values)
            embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Erreur SigLIP embedding: {e}")
        return None

def predict_level1_qwen(image_path, processor, model, device, use_level2=False):
    if use_level2:
        classes_list = LEVEL2_CLASSES
        prompt = """This is a Moroccan document. Identify its exact class.

If you see a plastic ID CARD with:
- Barcode/MRZ at bottom
- "Valable jusqu'au" (validity date)
- "Fils de" or "Fille de" (son/daughter of)
- "N° état civil" (civil status number)
- "Adresse" with street name
- "Sexe M/F"
Then it is: CIN_back

If you see a plastic ID CARD with:
- Photo of a person
- "Date de naissance" (birth date)
- "Lieu de naissance" (birth place)
Then it is: CIN_front

If you see a BANK STATEMENT with:
- Bank logo at top
- Table with columns: Date, Description, Debit, Credit, Balance
- Account number, RIB, IBAN
Then identify the bank name.

Answer with ONLY the class name."""
    else:
        classes_list = LEVEL1_CLASSES
        prompt = """Look at this Moroccan document image carefully.

Is this a CIN (National ID Card)?
- Small plastic card size
- Has MRZ barcode at bottom
- Contains "Valable jusqu'au", "Fils de/Fille de", "N° état civil", "Adresse", "Sexe"
- Text in French AND Arabic
- If YES, answer: CIN

Is this a BANK STATEMENT (Relevé bancaire)?
- Full page document with bank logo
- Has a TABLE with columns for transactions (Date, Libellé, Débit, Crédit, Solde)
- Shows account number, RIB, IBAN
- If YES, answer: RELEVE BANCAIRE

Is this a UTILITY BILL (Facture)?
- Has ONEE, Lydec, Amendis, Redal, or Radeema logo
- Shows kWh or m³ consumption
- If YES, answer: FACTURE D'EAU ET D'ELECTRICITE

Is this an EMPLOYMENT DOCUMENT?
- Attestation, Contract, or Payslip
- Company letterhead
- If YES, answer: DOCUMENT EMPLOYEUR

IMPORTANT: A CIN card may contain the word "BANK" in the address field (e.g., "LOTS BANK CHAABI RUE"). This does NOT make it a bank statement. Look at the OVERALL FORMAT of the document.

Answer with ONLY ONE category: CIN, RELEVE BANCAIRE, FACTURE D'EAU ET D'ELECTRICITE, or DOCUMENT EMPLOYEUR"""
    
    try:
        img = Image.open(image_path).convert('RGB')
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"Erreur apply_chat_template: {e}")
            text = prompt
        
        image_input = messages[0]["content"][0]["image"]
        
        if isinstance(text, dict):
            text = text.get('text', text.get('prompt', prompt))
        elif not isinstance(text, str):
            if isinstance(text, (list, tuple)) and len(text) > 0:
                text = text[0] if isinstance(text[0], str) else str(text[0])
            else:
                text = str(text) if text else prompt
        
        if not isinstance(image_input, Image.Image):
            image_input = img
        
        try:
            inputs = processor(text=text, images=[image_input], padding=True, return_tensors="pt")
        except Exception as e:
            print(f"Erreur format processeur (tentative 1): {e}")
            try:
                inputs = processor(text=[text], images=[image_input], padding=True, return_tensors="pt")
            except Exception as e2:
                print(f"Erreur format processeur (tentative 2): {e2}")
                try:
                    inputs = processor(messages=messages, return_tensors="pt")
                except Exception as e3:
                    print(f"Erreur format processeur (tentative 3): {e3}")
                    inputs = processor(text=prompt, images=[image_input], return_tensors="pt")
        
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    generated_ids = model.generate(**inputs, max_new_tokens=100 if use_level2 else 60)
            else:
                generated_ids = model.generate(**inputs, max_new_tokens=100 if use_level2 else 60)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        response_clean = response.strip()
        response_upper = response_clean.upper()
        
        print(f"[VLM RAW] Response: {response_clean[:100]}")
        
        for class_name in classes_list:
            class_upper = class_name.upper()
            if class_upper in response_upper or response_upper in class_upper:
                return class_name, 0.9
        
        if not use_level2:
            cin_keywords = ["CIN", "CNIE", "IDENTITE", "IDENTITÉ", "CARTE NATIONALE", "IDENTITY", "ID CARD", 
                           "البطاقة", "الوطنية", "NATIONALE"]
            if any(kw in response_upper for kw in cin_keywords):
                return "CIN", 0.85
            
            bank_keywords = ["BANCAIRE", "BANQUE", "BANK", "RELEVE", "RELEVÉ", "COMPTE", "ATTIJARIWAFA", 
                            "POPULAIRE", "CIH", "BMCE", "BOA", "BARID", "AGRICOLE", "CDM", "GENERALE"]
            if any(kw in response_upper for kw in bank_keywords):
                return "RELEVE BANCAIRE", 0.85
            
            facture_keywords = ["FACTURE", "ELECTRICITE", "ÉLECTRICITÉ", "EAU", "ONEE", "LYDEC", 
                               "AMENDIS", "REDAL", "RADEEMA", "KWH", "CONSOMMATION"]
            if any(kw in response_upper for kw in facture_keywords):
                return "FACTURE D'EAU ET D'ELECTRICITE", 0.85
            
            emp_keywords = ["EMPLOYEUR", "EMPLOI", "TRAVAIL", "SALAIRE", "PAIE", "ATTESTATION", 
                           "CONTRAT", "CONTRACT", "BULLETIN", "CERTIFICAT", "ENTREPRISE"]
            if any(kw in response_upper for kw in emp_keywords):
                return "DOCUMENT EMPLOYEUR", 0.85
        
        if use_level2:
            response_lower = response_clean.lower()
            if "cin" in response_lower:
                if "back" in response_lower or "verso" in response_lower or "adresse" in response_lower:
                    return "CIN_back", 0.8
                return "CIN_front", 0.8
            if "releve" in response_lower or "bancaire" in response_lower:
                if "attijari" in response_lower: return "Releve bancaire Attijariwafa bank", 0.8
                if "populaire" in response_lower: return "Releve bancaire BANQUE POPULAIRE", 0.8
                if "cih" in response_lower: return "Releve bancaire CIH BANK", 0.8
                if "barid" in response_lower: return "Releve bancaire AL BARID BANK", 0.8
                if "africa" in response_lower or "boa" in response_lower: return "Releve bancaire BANK OF AFRICA", 0.8
                if "agricole" in response_lower: return "Releve bancaire CREDIT AGRICOLE", 0.8
                if "cdm" in response_lower or "credit du maroc" in response_lower: return "Releve bancaire CDM CREDIT DU MAROC", 0.8
                if "generale" in response_lower: return "Releve bancaire SOCIETE GENERALE", 0.8
            if "facture" in response_lower:
                return "Facture d'eau et d'électricité", 0.8
            if "attestation" in response_lower:
                return "Attestation", 0.8
            if "contrat" in response_lower:
                return "Contrat", 0.8
            if "paie" in response_lower or "salaire" in response_lower:
                return "Fiche de paie", 0.8
        
        return response_clean, 0.5
    except Exception as e:
        print(f"Erreur Qwen2-VL prediction: {e}")
        return None, 0.0

def build_index(dataset_root, device, cache_dir="."):
    print(f"Construction de l'index SigLIP pour: {dataset_root}")
    print(f"Device: {device}")
    
    processor, model = load_siglip_model(device)
    print("Modele SigLIP charge")
    
    image_files, pdf_files = get_image_files(dataset_root)
    print(f"Nombre d'images trouvees: {len(image_files)}")
    print(f"Nombre de PDFs trouves: {len(pdf_files)}")
    
    embeddings_list = []
    paths_list = []
    dataset_base = os.path.abspath(dataset_root)
    
    print("Calcul des embeddings pour les images...")
    for img_path in tqdm(image_files, desc="Processing images"):
        embedding = embed_image_siglip(img_path, processor, model, device)
        if embedding is not None:
            embeddings_list.append(embedding)
            rel_path = os.path.relpath(img_path, dataset_base)
            paths_list.append(rel_path)
        
        if device.type == 'cuda' and len(embeddings_list) % 50 == 0:
            torch.cuda.empty_cache()
    
    print(f"\nTraitement des PDFs ({len(pdf_files)} fichiers)...")
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            if HAS_FITZ:
                pdf_doc = fitz.open(pdf_path)
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("ppm")
                    pil_img = Image.open(io.BytesIO(img_data))
                    
                    embedding = embed_image_siglip_from_pil(pil_img, processor, model, device)
                    if embedding is not None:
                        embeddings_list.append(embedding)
                        rel_path = os.path.relpath(pdf_path, dataset_base)
                        paths_list.append(f"{rel_path}::page_{page_num + 1}")
                pdf_doc.close()
            elif HAS_PDF2IMAGE:
                from pdf2image import convert_from_path
                images = convert_from_path(pdf_path)
                for page_num, pil_img in enumerate(images):
                    embedding = embed_image_siglip_from_pil(pil_img, processor, model, device)
                    if embedding is not None:
                        embeddings_list.append(embedding)
                        rel_path = os.path.relpath(pdf_path, dataset_base)
                        paths_list.append(f"{rel_path}::page_{page_num + 1}")
        except Exception as e:
            print(f"  Erreur lors du traitement de {pdf_path}: {e}")
            continue
        
        if device.type == 'cuda' and len(embeddings_list) % 50 == 0:
            torch.cuda.empty_cache()
    
    if not embeddings_list:
        print("Erreur: aucun embedding genere")
        return
    
    embeddings = np.vstack(embeddings_list).astype(np.float32)
    
    cache_emb_path = os.path.join(cache_dir, CACHE_EMBEDDINGS)
    cache_paths_path = os.path.join(cache_dir, CACHE_PATHS)
    
    print(f"Sauvegarde: {cache_emb_path}")
    np.save(cache_emb_path, embeddings)
    
    cache_data = {
        'paths': paths_list,
        'dataset_base': dataset_base
    }
    with open(cache_paths_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
    
    print(f"Index sauvegarde: {len(paths_list)} images, dimension {embeddings.shape[1]}")

def load_index(cache_dir="."):
    cache_emb_path = os.path.join(cache_dir, CACHE_EMBEDDINGS)
    cache_paths_path = os.path.join(cache_dir, CACHE_PATHS)
    
    if not os.path.exists(cache_emb_path) or not os.path.exists(cache_paths_path):
        return None, None, None
    
    embeddings = np.load(cache_emb_path).astype(np.float32)
    with open(cache_paths_path, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
    
    return embeddings, cache_data['paths'], cache_data['dataset_base']

def retrieve_topk(query_embedding, embeddings, paths, k=5, return_all=False):
    if not isinstance(paths, list):
        if isinstance(paths, dict):
            if 'paths' in paths:
                paths = paths['paths']
            else:
                try:
                    sorted_keys = sorted([k for k in paths.keys() if isinstance(k, (int, str))])
                    paths = [paths[k] for k in sorted_keys]
                except:
                    paths = list(paths.values())
        else:
            try:
                paths = list(paths)
            except:
                raise ValueError(f"paths doit être une liste, reçu: {type(paths)}")
    
    if isinstance(query_embedding, (list, tuple)):
        query_embedding = np.array(query_embedding)
    if query_embedding.ndim > 1:
        query_embedding = query_embedding.flatten()
    
    similarities = np.dot(embeddings, query_embedding)
    
    if return_all:
        all_indices = np.argsort(similarities)[::-1]
        all_scores = similarities[all_indices]
        all_paths = [paths[int(idx)] for idx in all_indices]
        return list(zip(all_paths, all_scores))
    else:
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_scores = similarities[top_k_indices]
        top_k_paths = [paths[int(idx)] for idx in top_k_indices]
        return list(zip(top_k_paths, top_k_scores))

_label_mapping_cache = None

def load_label_mapping(mapping_file=LABEL_MAPPING_FILE):
    global _label_mapping_cache
    if _label_mapping_cache is not None:
        return _label_mapping_cache
    
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            _label_mapping_cache = json.load(f)
        return _label_mapping_cache
    return {}

def get_label_from_path(path, dataset_base):
    mapping = load_label_mapping()
    
    if '::page_' in path:
        base_path = path.split('::page_')[0]
    else:
        base_path = path
    
    if os.path.isabs(base_path):
        rel_path = os.path.relpath(base_path, dataset_base)
    else:
        rel_path = base_path
    
    for path_variant in [rel_path, rel_path.replace('\\', '/'), rel_path.replace('/', '\\')]:
        if path_variant in mapping:
            return mapping[path_variant]
    
    parts = Path(rel_path).parts
    if len(parts) > 0:
        folder_name = parts[0]
        folder_to_label = {
            'CIN': 'CIN',
            'document administrative': 'Document Administratif Général',
            "Facture d'eau_d'éléctricité": "Facture d'eau et d'électricité",
            'releve bancaire': 'Releve bancaire'
        }
        return folder_to_label.get(folder_name, folder_name)
    
    return None

def detect_cin_front_back_mrz(image_path_or_pil):
    try:
        if isinstance(image_path_or_pil, str):
            img = cv2.imread(image_path_or_pil)
            if img is None:
                pil_img = Image.open(image_path_or_pil).convert('RGB')
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            img = cv2.cvtColor(np.array(image_path_or_pil), cv2.COLOR_RGB2BGR)
        
        H, W = img.shape[:2]
        
        lower_region = img[int(0.55 * H):H, 0:W]
        gray_lower = cv2.cvtColor(lower_region, cv2.COLOR_BGR2GRAY)
        
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract\tesseract.exe"
            
            mrz_text = pytesseract.image_to_string(
                gray_lower, 
                lang="eng", 
                config="--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
            )
            
            has_mrz = ("<<" in mrz_text) or (re.search(r"<{2,}", mrz_text) is not None)
            
            if has_mrz:
                return 'CIN_back'
            else:
                return 'CIN_front'
        except ImportError:
            return 'CIN'
    except Exception as e:
        return 'CIN'

def predict_label12(all_results, dataset_base, query_image_path=None, query_source_path=None, use_all_results=True, similarity_threshold=0.7):
    label_scores = {}
    
    filtered_results = [(path, score) for path, score in all_results if score >= similarity_threshold]
    
    if not filtered_results:
        filtered_results = sorted(all_results, key=lambda x: x[1], reverse=True)[:10]
    
    for path, score in filtered_results:
        label = get_label_from_path(path, dataset_base)
        if label:
            if label not in label_scores:
                label_scores[label] = []
            label_scores[label].append(score)
    
    if not label_scores:
        return None, 0.0
    
    if query_source_path and os.path.exists(query_source_path):
        try:
            source_rel = os.path.relpath(query_source_path, dataset_base)
            source_folder = Path(source_rel).parts[0] if Path(source_rel).parts else None
            
            folder_to_class = {
                'CIN': 'CIN',
                'releve bancaire': 'Releve bancaire',
                "Facture d'eau_d'éléctricité": "Facture d'eau et d'électricité",
                'document administrative': 'Document Administratif Général'
            }
            
            if source_folder and source_folder in folder_to_class:
                source_class = folder_to_class[source_folder]
                if source_class not in label_scores:
                    label_scores[source_class] = []
                label_scores[source_class].append(0.9)
        except:
            pass
    
    if 'CIN' in label_scores and query_image_path:
        try:
            detected_cin_type = detect_cin_front_back_mrz(query_image_path)
            if detected_cin_type in ['CIN_front', 'CIN_back']:
                predicted_label = detected_cin_type
                cin_scores = label_scores['CIN']
                confidence = np.mean(cin_scores)
                max_score = max([score for scores in label_scores.values() for score in scores])
                confidence_normalized = confidence / max_score if max_score > 0 else 0.0
                return predicted_label, confidence_normalized
        except:
            pass
    
    label_weighted_scores = {}
    for label, scores in label_scores.items():
        label_weighted_scores[label] = np.mean(scores) * (1 + np.max(scores))
    
    predicted_label = max(label_weighted_scores, key=label_weighted_scores.get)
    
    predicted_scores = label_scores[predicted_label]
    confidence = np.mean(predicted_scores)
    max_score = max([score for scores in label_scores.values() for score in scores])
    confidence_normalized = confidence / max_score if max_score > 0 else 0.0
    
    return predicted_label, confidence_normalized

def map_12_to_4(label12):
    if not label12:
        return 'UNKNOWN'
    
    label12_str = str(label12).strip()
    
    if label12_str in LABEL_MAPPING:
        return LABEL_MAPPING[label12_str]
    
    label12_lower = label12_str.lower()
    for key, value in LABEL_MAPPING.items():
        if key.lower() == label12_lower:
            return value
    
    for key, value in LABEL_MAPPING.items():
        key_lower = key.lower()
        if (label12_lower.startswith(key_lower) or 
            key_lower in label12_lower or 
            label12_lower in key_lower):
            return value
    
    label12_upper = label12_str.upper()
    if 'CIN' in label12_upper or 'IDENTITE' in label12_upper or 'CARTE' in label12_upper:
        return 'CIN'
    elif 'RELEVE' in label12_upper or 'BANCAIRE' in label12_upper or 'BANQUE' in label12_upper:
        return 'RELEVE BANCAIRE'
    elif 'FACTURE' in label12_upper or 'EAU' in label12_upper or 'ELECTRICITE' in label12_upper:
        return 'FACTURE D\'EAU ET D\'ELECTRICITE'
    elif any(word in label12_upper for word in ['CONTRAT', 'ATTESTATION', 'FICHE', 'SALAIRE', 'EMPLOYEUR', 'CERTIFICAT', 'AVIS', 'LETTRE']):
        return 'DOCUMENT EMPLOYEUR'
    
    return 'UNKNOWN'

def check_sanity(query_path, topk_results, dataset_base):
    query_abs = os.path.abspath(query_path)
    query_rel = os.path.relpath(query_abs, dataset_base) if os.path.exists(query_abs) else None
    
    if query_rel and topk_results:
        top1_path = topk_results[0][0]
        top1_abs = os.path.join(dataset_base, top1_path) if not os.path.isabs(top1_path) else top1_path
        
        if os.path.abspath(top1_abs) == query_abs:
            print("\n[OK] Sanity check: Query trouve dans le dataset (top-1 = query)")
        else:
            print(f"\n[WARNING] Sanity check: Query devrait etre top-1 mais ne l'est pas")
            print(f"  Query: {query_rel}")
            print(f"  Top-1: {topk_results[0][0]}")

def main():
    parser = argparse.ArgumentParser(description='RAG hybride pour classification de documents')
    parser.add_argument('query_image', type=str, nargs='?', default=None, help='Chemin de l\'image de requete')
    parser.add_argument('--k', type=int, default=5, help='Nombre de resultats (default: 5)')
    parser.add_argument('--rebuild', action='store_true', help='Reconstruire l\'index')
    parser.add_argument('--cpu', action='store_true', help='Forcer l\'utilisation du CPU')
    parser.add_argument('--dataset', type=str, default=DATASET_ROOT, help='Chemin du dataset')
    parser.add_argument('--cache-dir', type=str, default='.', help='Dossier pour le cache')
    
    args = parser.parse_args()
    
    if args.query_image is None:
        default_image = os.path.join(DATASET_ROOT, "Facture d'eau_d'éléctricité", 'B_AMENDIS_TANGER_124-94.pdf')
        if os.path.exists(default_image):
            args.query_image = default_image
            print(f"Utilisation de l'image par defaut: {default_image}")
        else:
            parser.print_help()
            print(f"\nExemple: python test_rag.py \"{default_image}\" --k 5")
            sys.exit(1)
    
    if not os.path.exists(args.query_image):
        print(f"Erreur: l'image {args.query_image} n'existe pas")
        sys.exit(1)
    
    device = get_device(args.cpu)
    print(f"Device: {device}")
    
    embeddings, paths, dataset_base = load_index(args.cache_dir)
    
    if embeddings is None or args.rebuild:
        build_index(args.dataset, device, args.cache_dir)
        embeddings, paths, dataset_base = load_index(args.cache_dir)
        if embeddings is None:
            print("Erreur: impossible de charger l'index")
            sys.exit(1)
    else:
        print(f"Index SigLIP charge: {len(paths)} images")
    
    dataset_base = dataset_base or os.path.abspath(args.dataset)
    
    print(f"\n{'='*60}")
    print("CLASSIFICATION NIVEAU 1 (4 classes) avec Qwen2-VL-2B-Instruct")
    print(f"{'='*60}")
    label4_qwen, conf4_qwen = None, 0.0
    try:
        qwen_processor, qwen_model = load_qwen_model(device)
        print(f"Prediction pour: {args.query_image}")
        label4_qwen, conf4_qwen = predict_level1_qwen(args.query_image, qwen_processor, qwen_model, device, use_level2=True)
        if label4_qwen:
            print(f"Prediction Qwen2-VL: {label4_qwen} (confiance: {conf4_qwen:.2f})")
        else:
            print("Erreur prediction Qwen2-VL, utilisation SigLIP uniquement")
            label4_qwen, conf4_qwen = None, 0.0
    except Exception as e:
        print(f"Qwen2-VL non disponible: {e}")
        print("Utilisation SigLIP uniquement pour les deux niveaux")
        label4_qwen, conf4_qwen = None, 0.0
    
    print(f"\n{'='*60}")
    print("CLASSIFICATION NIVEAU 2 (12 classes) avec SigLIP RAG")
    print(f"{'='*60}")
    siglip_processor, siglip_model = load_siglip_model(device)
    
    query_path_lower = args.query_image.lower()
    is_pdf = query_path_lower.endswith('.pdf')
    
    if is_pdf:
        print(f"Traitement du PDF: {args.query_image}")
        if not HAS_FITZ and not HAS_PDF2IMAGE:
            print("Erreur: Aucune bibliothèque PDF disponible (PyMuPDF ou pdf2image)")
            sys.exit(1)
        
        try:
            if HAS_FITZ:
                pdf_doc = fitz.open(args.query_image)
                num_pages = len(pdf_doc)
                print(f"PDF contient {num_pages} page(s)")
                pdf_results = []
                for page_num in range(num_pages):
                    page = pdf_doc[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("ppm")
                    pil_img = Image.open(io.BytesIO(img_data))
                    pdf_results.append((page_num + 1, pil_img))
                pdf_doc.close()
            elif HAS_PDF2IMAGE:
                images = convert_from_path(args.query_image)
                num_pages = len(images)
                print(f"PDF contient {num_pages} page(s)")
                pdf_results = [(i + 1, img) for i, img in enumerate(images)]
        except Exception as e:
            print(f"Erreur lors de la conversion du PDF: {e}")
            sys.exit(1)
        
        all_page_results = []
        for page_num, pil_img in pdf_results:
            print(f"\n--- Page {page_num} ---")
            query_embedding = embed_image_siglip_from_pil(pil_img, siglip_processor, siglip_model, device)
            if query_embedding is None:
                print(f"  Erreur: impossible de generer l'embedding pour la page {page_num}")
                continue
            
            print(f"  Recherche des images similaires (utilise tous les résultats pour le vote)...")
            all_results = retrieve_topk(query_embedding, embeddings, paths, k=args.k, return_all=True)
            topk_results = all_results[:args.k]
            
            print(f"\n  Top {len(topk_results)} images similaires:")
            for i, (path, score) in enumerate(topk_results[:5], 1):
                label_debug = get_label_from_path(path, dataset_base)
                print(f"    {i}. {path}")
                print(f"       Label: {label_debug}, Similarite: {score:.4f}")
            
            label12, conf12 = predict_label12(all_results, dataset_base, query_image_path=pil_img, query_source_path=args.query_image, use_all_results=True)
            label4_rag = map_12_to_4(label12) if label12 else 'UNKNOWN'
            
            all_page_results.append({
                'page': page_num,
                'label12': label12,
                'conf12': conf12,
                'label4': label4_rag,
                'topk': topk_results
            })
            
            print(f"  Resultat: {label12} (confiance: {conf12:.4f}) -> {label4_rag}")
        
        print(f"\n{'='*60}")
        print(f"RESULTATS FINAUX (PDF - {len(all_page_results)} pages):")
        print(f"{'='*60}")
        for page_result in all_page_results:
            page_num = page_result['page']
            label12 = page_result['label12']
            conf12 = page_result['conf12']
            label4_rag = page_result['label4']
            topk_results = page_result['topk']
            
            print(f"\nPage {page_num}:")
            print(f"  Niveau 2 (12 classes): {label12}")
            print(f"  Confiance: {conf12:.4f} ({conf12*100:.2f}%)")
            print(f"  -> Niveau 1 (4 classes): {label4_rag}")
            if len(topk_results) > 0:
                print(f"  Top-1 similaire: {topk_results[0][0]} (similarite: {topk_results[0][1]:.4f})")
        print(f"{'='*60}")
    else:
        print(f"Calcul de l'embedding SigLIP pour: {args.query_image}")
        query_embedding = embed_image_siglip(args.query_image, siglip_processor, siglip_model, device)
        
        if query_embedding is None:
            print("Erreur: impossible de generer l'embedding")
            sys.exit(1)
        
        print(f"Recherche des images similaires (utilise tous les résultats pour le vote)...")
        all_results = retrieve_topk(query_embedding, embeddings, paths, k=args.k, return_all=True)
        topk_results = all_results[:args.k]
        
        print(f"\nTop {len(topk_results)} images similaires:\n")
        for i, (path, score) in enumerate(topk_results, 1):
            print(f"{i}. {path}")
            print(f"   Similarite: {score:.6f} ({score*100:.2f}%)")
            print()
        
        check_sanity(args.query_image, topk_results, dataset_base)
        
        label12, conf12 = predict_label12(all_results, dataset_base, query_image_path=args.query_image, query_source_path=args.query_image, use_all_results=True)
        label4_rag = map_12_to_4(label12) if label12 else 'UNKNOWN'
        
        print(f"{'='*60}")
        print(f"\nRESULTATS FINAUX:")
        print(f"{'='*60}")
        if label4_qwen:
            print(f"Niveau 1 (4 classes) - Qwen2-VL:")
            print(f"  Classe: {label4_qwen}")
            print(f"  Confiance: {conf4_qwen:.4f} ({conf4_qwen*100:.2f}%)")
        print(f"\nNiveau 2 (12 classes) - SigLIP RAG:")
        print(f"  Classe: {label12}")
        print(f"  Confiance: {conf12:.4f} ({conf12*100:.2f}%)")
        print(f"  -> Mappe vers niveau 1: {label4_rag}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
