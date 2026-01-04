

import cv2
import numpy as np
import pickle
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
from PIL import Image
import io

try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    try:
        from pdf2image import convert_from_path
        HAS_PDF2IMAGE = True
    except ImportError:
        HAS_PDF2IMAGE = False

# Configuration
ORB_CACHE_DIR = "orb_cache"
ORB_GALLERY_FILE = os.path.join(ORB_CACHE_DIR, "orb_gallery.pkl")
ORB_REF_IMAGES_FILE = os.path.join(ORB_CACHE_DIR, "ref_images.json")
DEFAULT_N_REFS = 5  # Nombre d'images de référence par classe
ORB_MAX_FEATURES = 1000
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 8

# Extensions d'images supportées
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
PDF_EXTENSIONS = {'.pdf'}

# 4 grandes classes
CLASSES_4 = [
    "CIN",
    "DOCUMENT EMPLOYEUR",
    "FACTURE D'EAU ET D'ELECTRICITE",
    "RELEVE BANCAIRE"
]

# Mapping des dossiers vers les 4 grandes classes
FOLDER_TO_CLASS_4 = {
    'CIN': 'CIN',
    'releve bancaire': 'RELEVE BANCAIRE',
    "Facture d'eau_d'éléctricité": "FACTURE D'EAU ET D'ELECTRICITE",
    'document administrative': 'DOCUMENT EMPLOYEUR'
}

def get_orb_detector():
    """Crée et retourne un détecteur ORB configuré"""
    return cv2.ORB_create(
        nfeatures=ORB_MAX_FEATURES,
        scaleFactor=ORB_SCALE_FACTOR,
        nlevels=ORB_N_LEVELS
    )

def extract_orb_features(image_path: str) -> Optional[Tuple[List, np.ndarray]]:
    """
    Extrait les features ORB d'une image.
    
    Returns:
        Tuple (keypoints_list, descriptors) ou None si erreur
        keypoints_list: Liste de tuples (x, y, angle, response) pour chaque keypoint
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Essayer avec PIL si cv2 echoue (chemins avec caracteres speciaux)
            try:
                from PIL import Image
                pil_img = Image.open(image_path).convert('L')
                img = np.array(pil_img)
            except:
                return None
        
        if img is None or img.size == 0:
            return None
        
        orb = get_orb_detector()
        keypoints, descriptors = orb.detectAndCompute(img, None)
        
        if descriptors is None or len(descriptors) == 0:
            return None
        
        # Convertir les keypoints en liste serialisable: (x, y, angle, response)
        keypoints_list = [(kp.pt[0], kp.pt[1], kp.angle, kp.response) for kp in keypoints]
        
        return keypoints_list, descriptors
    except Exception as e:
        print(f"Erreur extraction ORB pour {image_path}: {e}")
        return None

def get_class_from_path(image_path: str, dataset_root: str) -> Optional[str]:
    """
    Determine la classe d'une image depuis son chemin.
    Utilise d'abord le mapping image_to_label_mapping.json, sinon le nom du dossier.
    """
    try:
        rel_path = os.path.relpath(image_path, dataset_root)
        
        # Note: On utilise maintenant les 4 grandes classes basées sur le nom du dossier
        # Le mapping exact depuis image_to_label_mapping.json n'est plus utilisé
        
        # Fallback: utiliser le nom du dossier
        parts = Path(rel_path).parts
        
        if len(parts) == 0:
            return None
        
        folder_name = parts[0]
        
        # Si c'est un PDF avec page, extraire le nom de base
        if '::page_' in folder_name:
            folder_name = folder_name.split('::page_')[0]
            folder_name = Path(folder_name).parts[0] if Path(folder_name).parts else folder_name
        
        # Mapping des dossiers vers les classes (par defaut)
        # Pour ORB, on utilise des classes generiques car on n'a pas de mapping exact
        folder_to_class_generic = {
            'CIN': 'CIN_front',  # Par defaut
            'releve bancaire': 'Releve bancaire BANQUE POPULAIRE',  # Par defaut
            "Facture d'eau_d'éléctricité": "Facture d'eau et d'électricité",
            'document administrative': 'Attestation'  # Par defaut
        }
        
        # Utiliser directement le mapping vers les 4 grandes classes
        class_4 = FOLDER_TO_CLASS_4.get(folder_name, None)
        if class_4 and class_4 in CLASSES_4:
            return class_4
        
        # Si aucune correspondance, retourner None
        return None
    except:
        return None

def extract_first_page_from_pdf(pdf_path: str, temp_dir: str = None) -> Optional[str]:
    """
    Extrait la première page d'un PDF comme image temporaire.
    
    Args:
        pdf_path: Chemin du PDF
        temp_dir: Dossier temporaire (si None, utilise le dossier du PDF)
    
    Returns:
        Chemin de l'image temporaire, ou None si erreur
    """
    try:
        if HAS_FITZ:
            pdf_doc = fitz.open(pdf_path)
            if len(pdf_doc) == 0:
                pdf_doc.close()
                return None
            
            page = pdf_doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("ppm")
            pdf_doc.close()
            
            # Sauvegarder temporairement
            import tempfile
            if temp_dir is None:
                temp_dir = os.path.dirname(pdf_path)
            
            # Créer un nom de fichier temporaire basé sur le PDF
            pdf_name = Path(pdf_path).stem
            temp_file = os.path.join(temp_dir, f"_orb_temp_{pdf_name}_page1.png")
            with open(temp_file, 'wb') as f:
                f.write(img_data)
            
            return temp_file
        elif HAS_PDF2IMAGE:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, first_page=1, last_page=1)
            if not images:
                return None
            
            import tempfile
            if temp_dir is None:
                temp_dir = os.path.dirname(pdf_path)
            
            pdf_name = Path(pdf_path).stem
            temp_file = os.path.join(temp_dir, f"_orb_temp_{pdf_name}_page1.png")
            images[0].save(temp_file)
            return temp_file
        else:
            return None
    except Exception as e:
        print(f"  Avertissement: Impossible d'extraire la premiere page de {pdf_path}: {e}")
        return None

def get_images_by_class(dataset_root: str, n_refs: int = None) -> Dict[str, List[str]]:
    """
    Récupère les images de référence par classe depuis le dataset.
    Si n_refs est None, utilise TOUS les fichiers disponibles.
    Pour les PDFs, extrait la première page comme image.
    
    Args:
        dataset_root: Racine du dataset
        n_refs: Nombre d'images par classe (None = toutes les images/PDFs)
    
    Returns:
        Dict {class_name: [list of image paths]}
    """
    images_by_class = {cls: [] for cls in CLASSES_4}
    temp_files = []  # Pour nettoyer après
    
    # Parcourir le dataset
    for root, _, files in os.walk(dataset_root):
        for file in files:
            file_path = os.path.join(root, file)
            suffix = Path(file).suffix.lower()
            
            # Traiter les images directement
            if suffix in IMAGE_EXTENSIONS:
                cls = get_class_from_path(file_path, dataset_root)
                
                if cls and cls in images_by_class:
                    # Si n_refs est None, ajouter toutes les images
                    # Sinon, limiter à n_refs
                    if n_refs is None or len(images_by_class[cls]) < n_refs:
                        images_by_class[cls].append(file_path)
            
            # Traiter les PDFs : extraire la première page
            elif suffix in PDF_EXTENSIONS:
                cls = get_class_from_path(file_path, dataset_root)
                
                if cls and cls in images_by_class:
                    if n_refs is None or len(images_by_class[cls]) < n_refs:
                        # Extraire la première page du PDF
                        temp_img = extract_first_page_from_pdf(file_path)
                        if temp_img:
                            images_by_class[cls].append(temp_img)
                            temp_files.append(temp_img)
    
    return images_by_class

def build_orb_gallery(dataset_root: str, n_refs: int = None, cache_dir: str = ORB_CACHE_DIR):
    """
    Construit la galerie ORB : extrait les features ORB pour les images de référence par classe.
    Si n_refs est None, utilise TOUS les fichiers disponibles.
    Sauvegarde en cache (pickle).
    """
    if n_refs is None:
        print(f"Construction de la galerie ORB avec TOUS les fichiers disponibles...")
    else:
        print(f"Construction de la galerie ORB avec {n_refs} images de référence par classe...")
    print(f"Dataset: {dataset_root}")
    
    # Créer le dossier de cache
    os.makedirs(cache_dir, exist_ok=True)
    
    # Récupérer les images par classe
    images_by_class = get_images_by_class(dataset_root, n_refs)
    
    orb = get_orb_detector()
    gallery = {}  # {class_name: [list of (keypoints, descriptors)]}
    ref_images_info = {}  # {class_name: [list of image paths]}
    
    for class_name in tqdm(CLASSES_4, desc="Classes"):
        image_paths = images_by_class[class_name]
        ref_images_info[class_name] = image_paths
        
        if not image_paths:
            print(f"  Avertissement: Aucune image trouvée pour la classe '{class_name}'")
            gallery[class_name] = []
            continue
        
        class_features = []
        for img_path in tqdm(image_paths, desc=f"  {class_name}", leave=False):
            features = extract_orb_features(img_path)
            if features is not None:
                keypoints, descriptors = features
                class_features.append((keypoints, descriptors))
        
        gallery[class_name] = class_features
        print(f"  {class_name}: {len(class_features)}/{len(image_paths)} images avec features valides")
    
    # Sauvegarder en cache
    cache_file = os.path.join(cache_dir, ORB_GALLERY_FILE.split(os.sep)[-1])
    with open(cache_file, 'wb') as f:
        pickle.dump(gallery, f)
    print(f"\nGalerie sauvegardée: {cache_file}")
    
    # Sauvegarder les chemins des images de référence
    ref_images_file = os.path.join(cache_dir, ORB_REF_IMAGES_FILE.split(os.sep)[-1])
    with open(ref_images_file, 'w', encoding='utf-8') as f:
        json.dump(ref_images_info, f, ensure_ascii=False, indent=2)
    print(f"Chemins des images de référence sauvegardés: {ref_images_file}")
    
    return gallery, ref_images_info

def load_orb_gallery(cache_dir: str = ORB_CACHE_DIR) -> Optional[Dict]:
    """Charge la galerie ORB depuis le cache"""
    cache_file = os.path.join(cache_dir, ORB_GALLERY_FILE.split(os.sep)[-1])
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            gallery = pickle.load(f)
        return gallery
    except Exception as e:
        print(f"Erreur lors du chargement de la galerie: {e}")
        return None

def match_orb_features(query_keypoints: List, query_descriptors: np.ndarray, 
                      ref_keypoints: List, ref_descriptors: np.ndarray, 
                      ratio_threshold: float = 0.75) -> Tuple[int, int]:
    """
    Match les descriptors ORB entre query et reference.
    Utilise BFMatcher + ratio test + RANSAC homography.
    
    Args:
        query_keypoints: Liste de tuples (x, y, angle, response) pour la query
        query_descriptors: Descriptors de la query
        ref_keypoints: Liste de tuples (x, y, angle, response) pour la reference
        ref_descriptors: Descriptors de la reference
        ratio_threshold: Seuil pour le ratio test
    
    Returns:
        Tuple (good_matches, inliers)
    """
    if query_descriptors is None or ref_descriptors is None:
        return 0, 0
    
    if len(query_descriptors) < 4 or len(ref_descriptors) < 4:
        return 0, 0
    
    # BFMatcher avec ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(query_descriptors, ref_descriptors, k=2)
    
    # Ratio test (Lowe's ratio test)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 4:
        return len(good_matches), 0
    
    # RANSAC homography pour trouver les inliers
    try:
        # Extraire les coordonnees des keypoints correspondants
        # query_keypoints est une liste de tuples (x, y, angle, response)
        src_pts = np.float32([[query_keypoints[m.queryIdx][0], query_keypoints[m.queryIdx][1]] 
                              for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([[ref_keypoints[m.trainIdx][0], ref_keypoints[m.trainIdx][1]] 
                              for m in good_matches]).reshape(-1, 1, 2)
        
        # RANSAC homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if mask is not None:
            inliers = int(mask.sum())
        else:
            inliers = len(good_matches)
    except Exception as e:
        # En cas d'erreur, utiliser tous les good_matches comme inliers
        inliers = len(good_matches)
    
    return len(good_matches), inliers

def compute_orb_score(query_features: Tuple, ref_features_list: List[Tuple], 
                      kp_query_count: int) -> Tuple[float, Dict]:
    """
    Calcule le score ORB pour une classe.
    
    Calcule le score pour chaque image de référence individuellement,
    puis prend le MEILLEUR score (pas la somme) pour éviter que les classes
    avec plus d'images aient un avantage.
    
    Score par image = 0.6 * inliers_ratio + 0.4 * (good_matches / max(1, kp_query))
    où inliers_ratio = inliers / max(1, good_matches)
    
    Score final = meilleur score parmi toutes les images de référence
    
    Returns:
        Tuple (score, debug_info)
    """
    if query_features is None or not ref_features_list:
        return 0.0, {}
    
    query_kp, query_desc = query_features
    
    best_score = 0.0
    best_debug = {"good_matches": 0, "inliers": 0, "kp_query": kp_query_count}
    total_good_matches = 0
    total_inliers = 0
    
    # Calculer le score pour chaque image de référence individuellement
    for ref_kp, ref_desc in ref_features_list:
        good_matches, inliers = match_orb_features(query_kp, query_desc, ref_kp, ref_desc)
        
        if good_matches > 0:
            inliers_ratio = inliers / max(1, good_matches)
            matches_ratio = good_matches / max(1, kp_query_count)
            img_score = 0.6 * inliers_ratio + 0.4 * matches_ratio
            
            # Garder le meilleur score
            if img_score > best_score:
                best_score = img_score
                best_debug = {
                    "good_matches": good_matches,
                    "inliers": inliers,
                    "kp_query": kp_query_count
                }
            
            # Garder aussi les totaux pour debug
            total_good_matches += good_matches
            total_inliers += inliers
    
    if best_score == 0.0:
        return 0.0, best_debug
    
    # Utiliser le meilleur score (déjà dans [0, 1])
    best_debug["total_good_matches"] = total_good_matches
    best_debug["total_inliers"] = total_inliers
    
    return best_score, best_debug

def match_query_orb(query_image_path: str, gallery: Dict, 
                   candidate_classes: Optional[List[str]] = None,
                   dataset_root: str = "") -> Dict:
    """
    Match une image query contre la galerie ORB.
    
    Args:
        query_image_path: Chemin de l'image query
        gallery: Galerie ORB chargée
        candidate_classes: Liste optionnelle de classes candidates (si None, teste toutes)
        dataset_root: Racine du dataset (pour debug)
    
    Returns:
        Dict avec best_class_12, best_score, scores_per_class, debug
    """
    # Extraire les features de la query
    query_features = extract_orb_features(query_image_path)
    
    if query_features is None:
        return {
            "best_class_4": None,
            "best_score": 0.0,
            "scores_per_class": {},
            "debug": {"error": "Impossible d'extraire les features ORB de la query"}
        }
    
    query_kp, query_desc = query_features
    kp_query_count = len(query_kp)
    
    # Déterminer les classes à tester
    classes_to_test = candidate_classes if candidate_classes else CLASSES_4
    
    scores_per_class = {}
    debug_info = {}
    
    for class_name in classes_to_test:
        if class_name not in gallery or not gallery[class_name]:
            scores_per_class[class_name] = 0.0
            continue
        
        score, class_debug = compute_orb_score(query_features, gallery[class_name], kp_query_count)
        scores_per_class[class_name] = score
        
        # Debug info pour toutes les classes avec score > 0
        if score > 0:
            debug_info[class_name] = class_debug
    
    # Trouver la meilleure classe
    if scores_per_class:
        best_class = max(scores_per_class, key=scores_per_class.get)
        best_score = scores_per_class[best_class]
    else:
        best_class = None
        best_score = 0.0
    
    return {
        "best_class_4": best_class,
        "best_score": best_score,
        "scores_per_class": scores_per_class,
        "debug": debug_info.get(best_class, {}) if best_class else {}
    }

def main():
    parser = argparse.ArgumentParser(description='ORB Matcher pour classification de documents')
    parser.add_argument('--query', type=str, help='Chemin de l\'image query')
    parser.add_argument('--dataset_root', type=str, default=r"C:\NLP-CV\dataset", 
                       help='Racine du dataset')
    parser.add_argument('--k', type=int, default=5, 
                       help='Nombre de classes candidates (si utilisé avec --use_candidates)')
    parser.add_argument('--use_candidates', type=str, default=None,
                       help='Liste de classes candidates séparées par virgule (ex: "CIN_front,Releve bancaire")')
    parser.add_argument('--build_gallery', action='store_true',
                       help='Construire la galerie ORB')
    parser.add_argument('--n_refs', type=int, default=None, nargs='?',
                       help='Nombre d\'images de référence par classe (défaut: toutes les images disponibles)')
    parser.add_argument('--cache_dir', type=str, default=ORB_CACHE_DIR,
                       help='Dossier pour le cache ORB')
    
    args = parser.parse_args()
    
    if args.build_gallery:
        # Construire la galerie
        gallery, ref_images = build_orb_gallery(args.dataset_root, args.n_refs, args.cache_dir)
        print(f"\nGalerie construite: {len(gallery)} classes")
        for cls, features_list in gallery.items():
            print(f"  {cls}: {len(features_list)} images de référence")
    elif args.query:
        # Charger la galerie
        gallery = load_orb_gallery(args.cache_dir)
        if gallery is None:
            print(f"Erreur: Galerie ORB non trouvée. Construisez-la d'abord avec --build_gallery")
            return
        
        # Déterminer les classes candidates
        candidate_classes = None
        if args.use_candidates:
            candidate_classes = [c.strip() for c in args.use_candidates.split(',')]
            print(f"Classes candidates: {candidate_classes}")
        else:
            print(f"Test de toutes les classes ({len(CLASSES_12)})")
        
        # Matcher la query
        result = match_query_orb(args.query, gallery, candidate_classes, args.dataset_root)
        
        print(f"\n{'='*60}")
        print(f"RESULTATS ORB pour: {args.query}")
        print(f"{'='*60}")
        print(f"Meilleure classe: {result['best_class_4']}")
        print(f"Score: {result['best_score']:.4f} ({result['best_score']*100:.2f}%)")
        
        if result['debug']:
            print(f"\nDebug:")
            for key, value in result['debug'].items():
                print(f"  {key}: {value}")
        
        print(f"\nScores par classe:")
        sorted_scores = sorted(result['scores_per_class'].items(), key=lambda x: x[1], reverse=True)
        for class_name, score in sorted_scores[:10]:  # Top 10
            if score > 0:
                print(f"  {class_name}: {score:.4f} ({score*100:.2f}%)")
        
        print(f"{'='*60}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

