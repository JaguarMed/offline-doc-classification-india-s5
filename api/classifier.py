# -*- coding: utf-8 -*-
import torch
import numpy as np
from PIL import Image
import io
import sys
import os
from pathlib import Path
import traceback
import time
import cv2
import re

sys.path.insert(0, str(Path(__file__).parent.parent / "PreProcessing"))

from api.config import *
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from PreProcessing.test_rag import (
    load_qwen_model, predict_level1_qwen,
    embed_image_siglip, retrieve_topk, predict_label12, get_label_from_path
)

def extract_pdf_pages(pdf_path):
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
    
    pages = []
    try:
        if HAS_FITZ:
            pdf_doc = fitz.open(pdf_path)
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("ppm")
                pil_img = Image.open(io.BytesIO(img_data))
                pages.append((page_num + 1, pil_img))
            pdf_doc.close()
            return pages
        elif HAS_PDF2IMAGE:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path)
            return [(i + 1, img) for i, img in enumerate(images)]
        else:
            print("Erreur: Aucune bibliothèque PDF disponible")
            return []
    except Exception as e:
        print(f"Erreur extraction PDF: {e}")
        return []

from PreProcessing.orb_matcher import load_orb_gallery, match_query_orb, CLASSES_4

def load_roberta_model(device):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ROBERTA)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ROBERTA)
        model.to(device)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Erreur chargement RoBERTa: {e}")
        return None, None

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Tesseract\tesseract.exe'
    HAS_OCR = True
except:
    HAS_OCR = False

CIN_KEYWORDS = {
    "strong": [
        "carte nationale", "carte d'identité", "البطاقة الوطنية",
        "royaume du maroc", "المملكة المغربية", "cnie",
        "carte nationale d'identité électronique",
        "identité nationale", "بطاقة التعريف الوطنية"
    ],
    "front": [
        "date de naissance", "تاريخ الازدياد", "née le", "né le",
        "lieu de naissance", "مكان الازدياد",
        "signature", "الإمضاء", "توقيع"
    ],
    "back": [
        "adresse", "العنوان", "address",
        "valable jusqu", "صالحة إلى غاية", "validité", "صالحة الى",
        "numéro de série", "رقم التسلسل",
        "date d'expiration", "تاريخ الانتهاء",
        "lieu de résidence", "محل السكنى",
        "fille de", "fils de", "بنت", "ابن",
        "n° état civil", "رقم الحالة المدنية", "etat civil",
        "sexe", "الجنس",
        "imm a", "lot", "temara", "rabat", "casa", "marrakech"
    ],
    "general": [
        "marocain", "marocaine", "مغربي", "مغربية",
        "nom", "prénom", "prenom", "الاسم", "النسب"
    ]
}

BANK_KEYWORDS = {
    "general": [
        "relevé de compte", "releve de compte", "relevé bancaire",
        "extrait de compte", "كشف حساب", "كشف الحساب",
        "solde créditeur", "solde débiteur", "الرصيد",
        "numéro de compte", "رقم الحساب", "rib", "iban",
        "date valeur", "libellé opération", "mouvement bancaire",
        "solde initial", "solde final", "nouveau solde",
        "virement reçu", "virement émis", "prélèvement", "versement espèces"
    ],
    "attijariwafa": ["attijariwafa bank", "attijari wafa", "التجاري وفا بنك"],
    "bank_of_africa": ["bank of africa", "bmce bank", "بنك أفريقيا"],
    "al_barid": ["al barid bank", "barid bank", "البريد بنك"],
    "banque_populaire": ["banque populaire", "البنك الشعبي"],
    "credit_agricole": ["crédit agricole du maroc", "credit agricole maroc", "القرض الفلاحي"],
    "cdm": ["crédit du maroc", "credit du maroc", "القرض المغربي"],
    "cih": ["cih bank", "التجاري الدولي للمغرب"],
    "societe_generale": ["société générale maroc", "societe generale maroc", "سوسيتي جنرال"]
}

FACTURE_KEYWORDS = {
    "general": [
        "facture", "فاتورة", "consommation", "استهلاك",
        "kwh", "m3", "كيلوواط", "متر مكعب",
        "compteur", "عداد", "index", "lecture",
        "montant ttc", "المبلغ الإجمالي", "total à payer",
        "redevance", "رسم", "taxe", "ضريبة",
        "période de consommation", "فترة الاستهلاك",
        "date limite", "تاريخ الاستحقاق",
        "référence client", "مرجع الزبون",
        "électricité", "كهرباء", "eau", "ماء"
    ],
    "onee": ["onee", "المكتب الوطني للكهرباء", "office national", "one"],
    "lydec": ["lydec", "ليديك", "casablanca"],
    "amendis": ["amendis", "أمانديس", "tanger", "tétouan"],
    "redal": ["redal", "ريضال", "rabat", "الرباط"],
    "radeema": ["radeema", "راديما", "marrakech", "مراكش"],
    "srm": ["srm", "الشركة الجهوية", "distribution"]
}

EMPLOYEUR_KEYWORDS = {
    "general": [
        "employeur", "المشغل", "salarié", "الأجير",
        "entreprise", "société", "شركة", "مقاولة",
        "fonction", "الوظيفة", "poste", "المنصب",
        "date d'embauche", "تاريخ التوظيف",
        "ancienneté", "الأقدمية",
        "cnss", "الصندوق الوطني للضمان الاجتماعي",
        "cimr", "الصندوق المهني المغربي للتقاعد",
        "matricule", "رقم التسجيل"
    ],
    "attestation": [
        "attestation de travail", "شهادة العمل",
        "attestation de salaire", "شهادة الأجر",
        "certifie que", "يشهد أن", "atteste que", "نشهد أن",
        "travaille au sein", "يعمل لدى",
        "à titre de", "بصفة", "en qualité de",
        "attestation d'emploi", "شهادة التشغيل"
    ],
    "contrat": [
        "contrat de travail", "عقد العمل", "عقد الشغل",
        "contrat à durée", "cdi", "cdd",
        "engagement", "التزام",
        "clause", "بند", "article", "المادة",
        "période d'essai", "فترة الاختبار",
        "durée du travail", "مدة العمل",
        "rémunération mensuelle", "الأجر الشهري",
        "obligations", "الالتزامات"
    ],
    "fiche_paie": [
        "bulletin de paie", "بيان الأجر",
        "fiche de paie", "ورقة الأجر",
        "salaire brut", "الأجر الإجمالي",
        "salaire net", "الأجر الصافي",
        "net à payer", "الصافي للأداء",
        "retenues", "الاقتطاعات",
        "cotisations", "المساهمات",
        "indemnités", "التعويضات",
        "heures travaillées", "ساعات العمل",
        "prime", "منحة", "bonus"
    ]
}


class DocumentClassifier:
    def __init__(self, use_cpu=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
        self.use_cpu = use_cpu
        self.qwen_processor = None
        self.qwen_model = None
        self.roberta_model = None
        self.roberta_tokenizer = None
        self.orb_gallery = None
        self.siglip_embeddings = None
        self.siglip_paths = None
        
    def load_models(self):
        try:
            print(f"Chargement modèles sur {self.device}...")
            self.qwen_processor, self.qwen_model = load_qwen_model(self.device)
            print("[OK] Qwen2-VL charge")
            
            try:
                if Path(MODEL_ROBERTA).exists():
                    self.roberta_model, self.roberta_tokenizer = load_roberta_model(self.device)
                    if self.roberta_model:
                        print("[OK] RoBERTa charge")
            except Exception as e:
                print(f"[WARN] RoBERTa non disponible: {e}")
            
            try:
                self.orb_gallery = load_orb_gallery(str(ORB_CACHE_DIR))
                if self.orb_gallery:
                    print("[OK] ORB gallery chargee")
                else:
                    print("[WARN] ORB gallery non trouvee")
            except Exception as e:
                print(f"[WARN] ORB non disponible: {e}")
            
            try:
                if EMBEDDINGS_CACHE.exists() and PATHS_CACHE.exists():
                    self.siglip_embeddings = np.load(str(EMBEDDINGS_CACHE))
                    with open(PATHS_CACHE, 'r') as f:
                        import json
                        cache_data = json.load(f)
                        if isinstance(cache_data, dict):
                            self.siglip_paths = cache_data.get('paths', [])
                        elif isinstance(cache_data, list):
                            self.siglip_paths = cache_data
                        else:
                            self.siglip_paths = []
                        if not isinstance(self.siglip_paths, list):
                            print(f"[WARN] siglip_paths n'est pas une liste: {type(self.siglip_paths)}")
                            self.siglip_paths = []
                    print("[OK] SigLIP index charge")
            except Exception as e:
                    print(f"[WARN] SigLIP index non disponible: {e}")
                
        except Exception as e:
            print(f"Erreur chargement modèles: {e}")
            traceback.print_exc()
    
    def extract_ocr_text(self, image_path):
        if not HAS_OCR:
            return None
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, lang="fra+ara+eng")
            return text.strip() if text.strip() else None
        except Exception as e:
            print(f"Erreur OCR: {e}")
            return None
    
    def _count_keyword_matches(self, text, keywords):
        if not text:
            return 0
        text_lower = text.lower()
        count = 0
        for kw in keywords:
            if kw.lower() in text_lower:
                count += 1
        return count
    
    def _detect_level1_from_ocr(self, ocr_text):
        if not ocr_text:
            return None, 0.0, {}
        
        text_lower = ocr_text.lower()
        scores = {}
        details = {}
        
        cin_strong = self._count_keyword_matches(ocr_text, CIN_KEYWORDS["strong"])
        cin_front = self._count_keyword_matches(ocr_text, CIN_KEYWORDS["front"])
        cin_back = self._count_keyword_matches(ocr_text, CIN_KEYWORDS["back"])
        cin_general = self._count_keyword_matches(ocr_text, CIN_KEYWORDS["general"])
        
        cin_bonus = 0
        if "état civil" in text_lower or "etat civil" in text_lower or "الحالة المدنية" in ocr_text:
            cin_bonus += 5
        if "valable jusqu" in text_lower or "صالحة" in ocr_text:
            cin_bonus += 5
        if "fils de" in text_lower or "fille de" in text_lower:
            cin_bonus += 4
        if "adresse" in text_lower and ("n°" in text_lower or "sexe" in text_lower):
            cin_bonus += 4
        
        cin_total = cin_strong * 4 + cin_front * 2 + cin_back * 2 + cin_general + cin_bonus
        scores["CIN"] = cin_total
        details["CIN"] = {"strong": cin_strong, "front": cin_front, "back": cin_back, "general": cin_general, "bonus": cin_bonus}
        
        bank_general = self._count_keyword_matches(ocr_text, BANK_KEYWORDS["general"])
        bank_specific = sum(
            self._count_keyword_matches(ocr_text, kws) 
            for key, kws in BANK_KEYWORDS.items() if key != "general"
        )
        
        bank_penalty = 0
        if cin_total >= 5:
            bank_penalty = bank_general
        
        bank_total = max(0, bank_general * 2 + bank_specific * 3 - bank_penalty * 2)
        scores["RELEVE BANCAIRE"] = bank_total
        details["RELEVE BANCAIRE"] = {"general": bank_general, "specific": bank_specific, "penalty": bank_penalty}
        
        facture_general = self._count_keyword_matches(ocr_text, FACTURE_KEYWORDS["general"])
        facture_specific = sum(
            self._count_keyword_matches(ocr_text, kws) 
            for key, kws in FACTURE_KEYWORDS.items() if key != "general"
        )
        facture_total = facture_general * 2 + facture_specific * 3
        scores["FACTURE D'EAU ET D'ELECTRICITE"] = facture_total
        details["FACTURE D'EAU ET D'ELECTRICITE"] = {"general": facture_general, "specific": facture_specific}
        
        emp_general = self._count_keyword_matches(ocr_text, EMPLOYEUR_KEYWORDS["general"])
        emp_attestation = self._count_keyword_matches(ocr_text, EMPLOYEUR_KEYWORDS["attestation"])
        emp_contrat = self._count_keyword_matches(ocr_text, EMPLOYEUR_KEYWORDS["contrat"])
        emp_paie = self._count_keyword_matches(ocr_text, EMPLOYEUR_KEYWORDS["fiche_paie"])
        emp_total = emp_general + emp_attestation * 2 + emp_contrat * 2 + emp_paie * 2
        scores["DOCUMENT EMPLOYEUR"] = emp_total
        details["DOCUMENT EMPLOYEUR"] = {"general": emp_general, "attestation": emp_attestation, "contrat": emp_contrat, "paie": emp_paie}
        
        print(f"[OCR SCORES] CIN={cin_total}, BANK={bank_total}, FACTURE={facture_total}, EMP={emp_total}")
        
        if not scores or max(scores.values()) == 0:
            return None, 0.0, details
        
        best_class = max(scores, key=scores.get)
        best_score = scores[best_class]
        
        if best_score < 3:
            return None, 0.0, details
        
        confidence = min(0.95, 0.5 + (best_score / 50.0))
        
        return best_class, confidence, details
    
    def _detect_level2_from_ocr(self, ocr_text, level1_class):
        if not ocr_text or not level1_class:
            return None, 0.0
        
        text_lower = ocr_text.lower()
        
        if level1_class == "CIN":
            text_lower = ocr_text.lower()
            front_score = 0
            back_score = 0
            
            if "royaume du maroc" in text_lower or "المملكة المغربية" in ocr_text:
                front_score += 5
            if "carte nationale" in text_lower or "البطاقة الوطنية" in ocr_text:
                front_score += 5
            if "né le" in text_lower or "née le" in text_lower:
                front_score += 5
            if "date de naissance" in text_lower or "تاريخ الازدياد" in ocr_text or "مزداد" in ocr_text:
                front_score += 5
            if "lieu de naissance" in text_lower or "مكان الازدياد" in ocr_text:
                front_score += 4
            if "tanger" in text_lower or "casablanca" in text_lower or "rabat" in text_lower or "marrakech" in text_lower:
                if "né" in text_lower or "naissance" in text_lower:
                    front_score += 3
            
            has_mrz = "<<" in ocr_text or re.search(r"<{2,}", ocr_text)
            if has_mrz:
                back_score += 15
            
            if "valable jusqu" in text_lower or "صالحة إلى" in ocr_text or "صالحة الى" in ocr_text:
                back_score += 8
            if "n° état civil" in text_lower or "n° etat civil" in text_lower or "رقم الحالة المدنية" in ocr_text:
                back_score += 6
            if "fille de" in text_lower or "fils de" in text_lower:
                back_score += 5
            if "adresse" in text_lower and ("lot" in text_lower or "rue" in text_lower or "imm" in text_lower):
                back_score += 6
            if "sexe" in text_lower and ("m" in text_lower or "f" in text_lower):
                back_score += 4
            
            if ("royaume" in text_lower or "المملكة" in ocr_text) and not has_mrz:
                front_score += 8
            
            print(f"[DEBUG CIN] front_score={front_score}, back_score={back_score}, has_mrz={has_mrz}")
            
            if back_score > front_score + 3:
                return "CIN_back", 0.90
            elif front_score > back_score:
                return "CIN_front", 0.90
            elif front_score > 0:
                return "CIN_front", 0.85
            return "CIN_front", 0.6
        
        if level1_class == "RELEVE BANCAIRE":
            bank_scores = {}
            bank_mapping = {
                "attijariwafa": "Releve bancaire Attijariwafa bank",
                "bank_of_africa": "Releve bancaire BANK OF AFRICA",
                "al_barid": "Releve bancaire AL BARID BANK",
                "banque_populaire": "Releve bancaire BANQUE POPULAIRE",
                "credit_agricole": "Releve bancaire CREDIT AGRICOLE",
                "cdm": "Releve bancaire CDM CREDIT DU MAROC",
                "cih": "Releve bancaire CIH BANK",
                "societe_generale": "Releve bancaire SOCIETE GENERALE"
            }
            
            for key, class_name in bank_mapping.items():
                score = self._count_keyword_matches(ocr_text, BANK_KEYWORDS.get(key, []))
                if score > 0:
                    bank_scores[class_name] = score
            
            if bank_scores:
                best_bank = max(bank_scores, key=bank_scores.get)
                return best_bank, 0.85
            return "Releve bancaire", 0.6
        
        if level1_class == "FACTURE D'EAU ET D'ELECTRICITE":
            return "Facture d'eau et d'électricité", 0.9
        
        if level1_class == "DOCUMENT EMPLOYEUR":
            attestation_score = self._count_keyword_matches(ocr_text, EMPLOYEUR_KEYWORDS["attestation"])
            contrat_score = self._count_keyword_matches(ocr_text, EMPLOYEUR_KEYWORDS["contrat"])
            paie_score = self._count_keyword_matches(ocr_text, EMPLOYEUR_KEYWORDS["fiche_paie"])
            
            scores = {
                "Attestation": attestation_score,
                "Contrat": contrat_score,
                "Fiche de paie": paie_score
            }
            
            if max(scores.values()) > 0:
                best_type = max(scores, key=scores.get)
                return best_type, 0.85
            return "Attestation", 0.6
        
        return level1_class, 0.5
    
    def classify_vlm(self, image_path, is_level1=True):
        try:
            start_time = time.time()
            result, confidence = predict_level1_qwen(
                str(image_path),
                self.qwen_processor,
                self.qwen_model,
                self.device,
                use_level2=not is_level1
            )
            elapsed = time.time() - start_time
            
            if is_level1:
                return {
                    "class": result,
                    "confidence": confidence,
                    "model": "VLM",
                    "elapsed": elapsed,
                    "raw_output": result
                }
            else:
                return self._classify_level2_rag(image_path, elapsed, fallback_class=result)
        except Exception as e:
            print(f"Erreur VLM: {e}")
            traceback.print_exc()
            return None
    
    def _detect_cin_side_vlm(self, image_path):
        try:
            from PIL import Image as PILImage
            img = PILImage.open(image_path).convert('RGB')
            
            prompt = """Look at this Moroccan ID card (CIN) image.

Is there a PHOTO of a person visible on the card?
- If YES (you can see a face/photo) → answer: FRONT
- If NO (no photo, but you see a barcode/MRZ at the bottom) → answer: BACK

Answer with ONLY one word: FRONT or BACK"""

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
                text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                text = prompt
            
            if isinstance(text, dict):
                text = text.get('text', prompt)
            elif not isinstance(text, str):
                text = str(text) if text else prompt
            
            try:
                inputs = self.qwen_processor(text=text, images=[img], padding=True, return_tensors="pt")
            except:
                inputs = self.qwen_processor(text=prompt, images=[img], return_tensors="pt")
            
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                if self.device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=10)
                else:
                    generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=10)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            response = self.qwen_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip().upper()
            
            print(f"[VLM CIN SIDE] Response: {response}")
            
            if "FRONT" in response:
                return "CIN_front", 0.95
            elif "BACK" in response:
                return "CIN_back", 0.95
            else:
                if "PHOTO" in response or "FACE" in response or "PERSON" in response:
                    return "CIN_front", 0.85
                elif "BARCODE" in response or "MRZ" in response:
                    return "CIN_back", 0.85
            
            return None, 0.0
        except Exception as e:
            print(f"Erreur VLM CIN side detection: {e}")
            return None, 0.0
    
    def _classify_level2_rag(self, image_path, vlm_time, fallback_class=None):
        try:
            if self.siglip_embeddings is None or self.siglip_paths is None:
                return {
                    "class": fallback_class or "UNKNOWN",
                    "confidence": 0.5,
                    "model": "VLM_FALLBACK",
                    "elapsed": vlm_time,
                    "raw_output": fallback_class
                }
            
            from PreProcessing.test_rag import load_siglip_model
            siglip_processor, siglip_model = load_siglip_model(self.device)
            query_emb = embed_image_siglip(str(image_path), siglip_processor, siglip_model, self.device)
            if query_emb is None:
                return {
                    "class": fallback_class or "UNKNOWN",
                    "confidence": 0.5,
                    "model": "VLM_FALLBACK",
                    "elapsed": vlm_time,
                    "raw_output": fallback_class
                }
            
            if not isinstance(self.siglip_paths, list):
                print(f"[WARN] siglip_paths n'est pas une liste avant retrieve_topk: {type(self.siglip_paths)}")
                if isinstance(self.siglip_paths, dict):
                    if 'paths' in self.siglip_paths:
                        self.siglip_paths = self.siglip_paths['paths']
                    else:
                        sorted_keys = sorted([k for k in self.siglip_paths.keys() if isinstance(k, (int, str))])
                        self.siglip_paths = [self.siglip_paths[k] for k in sorted_keys]
                else:
                    try:
                        self.siglip_paths = list(self.siglip_paths)
                    except:
                        print("[ERROR] Impossible de convertir siglip_paths en liste")
                        return {
                            "class": fallback_class or "UNKNOWN",
                            "confidence": 0.5,
                            "model": "VLM_FALLBACK",
                            "elapsed": vlm_time,
                            "raw_output": fallback_class
                        }
            
            topk_results = retrieve_topk(
                query_emb,
                self.siglip_embeddings,
                self.siglip_paths,
                k=20
            )
            
            label12, confidence = predict_label12(topk_results, DATASET_ROOT, query_image_path=str(image_path))
            
            return {
                "class": label12 or fallback_class or "UNKNOWN",
                "confidence": confidence,
                "model": "VLM_RAG",
                "elapsed": vlm_time,
                "raw_output": label12,
                "topk": topk_results[:5]
            }
        except Exception as e:
            print(f"Erreur RAG Level2: {e}")
            traceback.print_exc()
            return {
                "class": fallback_class or "UNKNOWN",
                "confidence": 0.5,
                "model": "VLM_FALLBACK",
                "elapsed": vlm_time,
                "raw_output": fallback_class
            }
    
    def classify_orb(self, image_path):
        try:
            if self.orb_gallery is None:
                return None
            
            start_time = time.time()
            result = match_query_orb(
                str(image_path),
                self.orb_gallery,
                candidate_classes=None,
                dataset_root=DATASET_ROOT
            )
            elapsed = time.time() - start_time
            
            if result and result.get("best_class_4"):
                return {
                    "class": result["best_class_4"],
                    "confidence": result["best_score"],
                    "model": "ORB",
                    "elapsed": elapsed,
                    "raw_output": result,
                    "scores_per_class": result.get("scores_per_class", {}),
                    "debug": result.get("debug", {})
                }
            return None
        except Exception as e:
            print(f"Erreur ORB: {e}")
            traceback.print_exc()
            return None
    
    def classify_roberta(self, text):
        try:
            if not text or self.roberta_model is None or self.roberta_tokenizer is None:
                return None
            
            start_time = time.time()
            
            inputs = self.roberta_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.roberta_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            id2label = self.roberta_model.config.id2label if hasattr(self.roberta_model.config, 'id2label') and self.roberta_model.config.id2label else None
            
            if id2label:
                predicted_idx = torch.argmax(probs, dim=-1).item()
                predicted_class = id2label[predicted_idx]
                confidence = float(probs[0][predicted_idx])
                
                num_classes = probs.shape[1]
                k = min(5, num_classes)
                top_probs, top_indices = torch.topk(probs[0], k)
                top_classes = [
                    {"class": id2label[idx.item()], "confidence": float(prob)} 
                    for prob, idx in zip(top_probs, top_indices)
                ]
            else:
                import pandas as pd
                dataset_file = Path(__file__).parent.parent / "dataset" / "Data.xlsx"
                if dataset_file.exists():
                    df = pd.read_excel(dataset_file)
                    classes = df['output'].unique().tolist()
                else:
                    classes = LEVEL1_CLASSES
                
                predicted_idx = torch.argmax(probs, dim=-1).item()
                predicted_class = classes[predicted_idx] if predicted_idx < len(classes) else LEVEL1_CLASSES[0]
                confidence = float(probs[0][predicted_idx])
                
                top_probs, top_indices = torch.topk(probs[0], min(5, len(classes)))
                top_classes = [
                    {"class": classes[idx.item()] if idx.item() < len(classes) else "UNKNOWN", "confidence": float(prob)} 
                    for prob, idx in zip(top_probs, top_indices)
                ]
            
            elapsed = time.time() - start_time
            
            return {
                "class": predicted_class,
                "confidence": confidence,
                "model": "RoBERTa",
                "elapsed": elapsed,
                "raw_output": {"predicted_class": predicted_class, "confidence": confidence},
                "top_classes": top_classes
            }
        except Exception as e:
            print(f"Erreur RoBERTa: {e}")
            traceback.print_exc()
            return None
    
    def fuse_results(self, vlm_result, orb_result, roberta_result, ocr_text=None):
        vlm_class = vlm_result.get("class") if vlm_result else None
        vlm_conf = vlm_result.get("confidence", 0.0) if vlm_result else 0.0
        
        orb_class = orb_result.get("class") if orb_result else None
        orb_conf = orb_result.get("confidence", 0.0) if orb_result else 0.0
        
        roberta_class = roberta_result.get("class") if roberta_result else None
        roberta_conf = roberta_result.get("confidence", 0.0) if roberta_result else 0.0
        
        ocr_class, ocr_conf, ocr_details = self._detect_level1_from_ocr(ocr_text)
        
        print(f"[FUSION] VLM={vlm_class}({vlm_conf:.2f}), ORB={orb_class}({orb_conf:.2f}), RoBERTa={roberta_class}({roberta_conf:.2f}), OCR={ocr_class}({ocr_conf:.2f})")
        
        votes = {}
        if vlm_class and vlm_class in LEVEL1_CLASSES:
            votes[vlm_class] = votes.get(vlm_class, 0) + vlm_conf
        if orb_class and orb_class in LEVEL1_CLASSES:
            votes[orb_class] = votes.get(orb_class, 0) + orb_conf + 0.2
        if roberta_class and roberta_class in LEVEL1_CLASSES:
            votes[roberta_class] = votes.get(roberta_class, 0) + roberta_conf + 0.1
        if ocr_class and ocr_class in LEVEL1_CLASSES:
            votes[ocr_class] = votes.get(ocr_class, 0) + ocr_conf + 0.3
        
        if orb_class and roberta_class and orb_class == roberta_class and orb_class != vlm_class:
            print(f"[FUSION] ORB + RoBERTa d'accord ({orb_class}), surpassent VLM ({vlm_class})")
            best_class = orb_class
            final_confidence = max(orb_conf, roberta_conf) + 0.1
            final_confidence = min(0.98, final_confidence)
            
            return {
                "level1": best_class,
                "confidence": final_confidence,
                "scores": {cls: final_confidence if cls == best_class else 0.0 for cls in LEVEL1_CLASSES},
                "models_used": {"vlm": True, "orb": True, "roberta": True},
                "detection_method": "ORB_ROBERTA_CONSENSUS",
                "ocr_details": ocr_details if ocr_class else {}
            }
        
        if ocr_class and ocr_conf >= 0.5:
            agreements = 0
            if orb_class == ocr_class: agreements += 1
            if roberta_class == ocr_class: agreements += 1
            if vlm_class == ocr_class: agreements += 1
            
            if agreements >= 1:
                print(f"[FUSION] OCR ({ocr_class}) confirmé par {agreements} modèle(s)")
                final_confidence = min(0.98, ocr_conf + 0.1 * agreements)
                return {
                    "level1": ocr_class,
                    "confidence": final_confidence,
                    "scores": {cls: final_confidence if cls == ocr_class else 0.0 for cls in LEVEL1_CLASSES},
                    "models_used": {"vlm": vlm_result is not None, "orb": orb_result is not None, "roberta": roberta_result is not None},
                    "detection_method": "OCR_CONFIRMED",
                    "ocr_details": ocr_details
                }
        
        if ocr_class and ocr_conf >= 0.75:
            print(f"[FUSION] OCR haute confiance ({ocr_class}, {ocr_conf:.2f})")
            return {
                "level1": ocr_class,
                "confidence": ocr_conf,
                "scores": {cls: ocr_conf if cls == ocr_class else 0.0 for cls in LEVEL1_CLASSES},
                "models_used": {"vlm": vlm_result is not None, "orb": orb_result is not None, "roberta": roberta_result is not None},
                "detection_method": "OCR_HIGH_CONF",
                "ocr_details": ocr_details
            }
        
        if votes:
            best_class = max(votes, key=votes.get)
            final_confidence = min(0.95, votes[best_class] / 2)
            print(f"[FUSION] Vote majoritaire: {best_class} (score={votes[best_class]:.2f})")
        else:
            best_class = vlm_class or "UNKNOWN"
            final_confidence = vlm_conf if vlm_class else 0.5
        
        level1_scores = {cls: 0.0 for cls in LEVEL1_CLASSES}
        if best_class in level1_scores:
            level1_scores[best_class] = final_confidence
        
        return {
            "level1": best_class,
            "confidence": final_confidence,
            "scores": level1_scores,
            "models_used": {
                "vlm": vlm_result is not None,
                "orb": orb_result is not None,
                "roberta": roberta_result is not None
            },
            "detection_method": "VOTE_MAJORITY",
            "ocr_details": ocr_details if ocr_class else {}
        }
    
    def classify_document(self, file_path, is_pdf=False):
        results = {
            "pages": [],
            "final_level1": None,
            "final_level2": None,
            "final_confidence": 0.0,
            "models_results": {},
            "errors": []
        }
        
        try:
            if is_pdf:
                pages = extract_pdf_pages(str(file_path))
                for page_num, pil_img in pages:
                    temp_path = UPLOAD_DIR / f"temp_page_{page_num}.png"
                    pil_img.save(temp_path)
                    
                    page_result = self._classify_single_image(temp_path)
                    page_result["page"] = page_num
                    results["pages"].append(page_result)
                    
                    try:
                        temp_path.unlink()
                    except:
                        pass
                
                if results["pages"]:
                    level1_votes = {}
                    level2_votes = {}
                    confidences = []
                    
                    for page in results["pages"]:
                        if page.get("final_level1"):
                            l1 = page["final_level1"]
                            level1_votes[l1] = level1_votes.get(l1, 0) + 1
                        if page.get("final_level2"):
                            l2 = page["final_level2"]
                            level2_votes[l2] = level2_votes.get(l2, 0) + 1
                        if page.get("final_confidence"):
                            confidences.append(page["final_confidence"])
                    
                    results["final_level1"] = max(level1_votes, key=level1_votes.get) if level1_votes else None
                    results["final_level2"] = max(level2_votes, key=level2_votes.get) if level2_votes else None
                    results["final_confidence"] = np.mean(confidences) if confidences else 0.0
            else:
                page_result = self._classify_single_image(file_path)
                page_result["page"] = 1
                results["pages"].append(page_result)
                results["final_level1"] = page_result.get("final_level1")
                results["final_level2"] = page_result.get("final_level2")
                results["final_confidence"] = page_result.get("final_confidence", 0.0)
            
            results["models_results"] = {
                "vlm_available": self.qwen_model is not None,
                "orb_available": self.orb_gallery is not None,
                "roberta_available": self.roberta_model is not None
            }
            
        except Exception as e:
            error_msg = f"Erreur classification: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            results["errors"].append(error_msg)
        
        return results
    
    def _classify_single_image(self, image_path):
        result = {
            "vlm_level1": None,
            "vlm_level2": None,
            "orb": None,
            "roberta": None,
            "ocr_text": None,
            "final_level1": None,
            "final_level2": None,
            "final_confidence": 0.0
        }
        
        ocr_text = self.extract_ocr_text(image_path)
        result["ocr_text"] = ocr_text[:500] if ocr_text else None
        
        ocr_predetect, ocr_preconf, _ = self._detect_level1_from_ocr(ocr_text)
        print(f"[DEBUG] OCR pré-détection: {ocr_predetect} (conf: {ocr_preconf:.2f})")
        
        vlm_l1 = self.classify_vlm(image_path, is_level1=True)
        result["vlm_level1"] = vlm_l1
        
        vlm_l2 = self.classify_vlm(image_path, is_level1=False)
        result["vlm_level2"] = vlm_l2
        
        skip_orb = (ocr_predetect == "DOCUMENT EMPLOYEUR" and ocr_preconf >= 0.8)
        
        if not skip_orb:
            orb_res = self.classify_orb(image_path)
            result["orb"] = orb_res
            print(f"[DEBUG] ORB result: {orb_res.get('class') if orb_res else 'None'}")
        else:
            result["orb"] = None
            orb_res = None
            print(f"[DEBUG] ORB skipped (DOCUMENT EMPLOYEUR detected by OCR)")
        
        if ocr_text:
            roberta_res = self.classify_roberta(ocr_text)
            result["roberta"] = roberta_res
        else:
            result["roberta"] = None
            roberta_res = None
        
        fusion_l1 = self.fuse_results(vlm_l1, orb_res, result["roberta"], ocr_text=ocr_text)
        result["final_level1"] = fusion_l1["level1"]
        result["final_confidence"] = fusion_l1["confidence"]
        
        if result["final_level1"] == "CIN":
            vlm_cin_side, vlm_cin_conf = self._detect_cin_side_vlm(image_path)
            if vlm_cin_side and vlm_cin_conf >= 0.8:
                result["final_level2"] = vlm_cin_side
                print(f"[CIN SIDE] VLM detected: {vlm_cin_side} (conf: {vlm_cin_conf:.2f})")
            else:
                ocr_l2, ocr_l2_conf = self._detect_level2_from_ocr(ocr_text, result["final_level1"])
                result["final_level2"] = ocr_l2 or "CIN_front"
                print(f"[CIN SIDE] OCR fallback: {result['final_level2']}")
        else:
            ocr_l2, ocr_l2_conf = self._detect_level2_from_ocr(ocr_text, result["final_level1"])
            
            if ocr_l2 and ocr_l2_conf >= 0.8:
                result["final_level2"] = ocr_l2
            elif vlm_l2 and vlm_l2.get("class"):
                result["final_level2"] = vlm_l2["class"]
            else:
                result["final_level2"] = ocr_l2 or result["final_level1"]
        
        return result


_classifier = None

def get_classifier(use_cpu=False):
    global _classifier
    if _classifier is None:
        _classifier = DocumentClassifier(use_cpu=use_cpu)
        _classifier.load_models()
    return _classifier
