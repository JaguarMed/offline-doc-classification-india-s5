

import cv2
import numpy as np
import pickle
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json

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

# Classes exactes (12 classes)
CLASSES_12 = [
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
        
        # Essayer de charger le mapping exact depuis image_to_label_mapping.json
        mapping_file = "image_to_label_mapping.json"
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    label_mapping = json.load(f)
                
                # Essayer avec differents formats de chemin
                for path_variant in [rel_path, rel_path.replace('\\', '/'), rel_path.replace('/', '\\')]:
                    if path_variant in label_mapping:
                        mapped_label = label_mapping[path_variant]
                        # Verifier que le label mappe est dans CLASSES_12
                        if mapped_label in CLASSES_12:
                            return mapped_label
            except:
                pass
        
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
        
        generic_class = folder_to_class_generic.get(folder_name, None)
        if generic_class and generic_class in CLASSES_12:
            return generic_class
        
        # Si aucune correspondance, retourner None
        return None
    except:
        return None

def get_images_by_class(dataset_root: str, n_refs: int = DEFAULT_N_REFS) -> Dict[str, List[str]]:
    """
    Récupère N images de référence par classe depuis le dataset.
    
    Returns:
        Dict {class_name: [list of image paths]}
    """
    images_by_class = {cls: [] for cls in CLASSES_12}
    
    # Parcourir le dataset
    for root, _, files in os.walk(dataset_root):
        for file in files:
            if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                file_path = os.path.join(root, file)
                cls = get_class_from_path(file_path, dataset_root)
                
                if cls and cls in images_by_class:
                    if len(images_by_class[cls]) < n_refs:
                        images_by_class[cls].append(file_path)
    
    return images_by_class

def build_orb_gallery(dataset_root: str, n_refs: int = DEFAULT_N_REFS, cache_dir: str = ORB_CACHE_DIR):
    """
    Construit la galerie ORB : extrait les features ORB pour N images de référence par classe.
    Sauvegarde en cache (pickle).
    """
    print(f"Construction de la galerie ORB avec {n_refs} images de référence par classe...")
    print(f"Dataset: {dataset_root}")
    
    # Créer le dossier de cache
    os.makedirs(cache_dir, exist_ok=True)
    
    # Récupérer les images par classe
    images_by_class = get_images_by_class(dataset_root, n_refs)
    
    orb = get_orb_detector()
    gallery = {}  # {class_name: [list of (keypoints, descriptors)]}
    ref_images_info = {}  # {class_name: [list of image paths]}
    
    for class_name in tqdm(CLASSES_12, desc="Classes"):
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
    
    Score = 0.6 * inliers_ratio + 0.4 * (good_matches / max(1, kp_query))
    où inliers_ratio = inliers / max(1, good_matches)
    
    Returns:
        Tuple (score, debug_info)
    """
    if query_features is None or not ref_features_list:
        return 0.0, {}
    
    query_kp, query_desc = query_features
    
    total_good_matches = 0
    total_inliers = 0
    
    for ref_kp, ref_desc in ref_features_list:
        good_matches, inliers = match_orb_features(query_kp, query_desc, ref_kp, ref_desc)
        total_good_matches += good_matches
        total_inliers += inliers
    
    if total_good_matches == 0:
        return 0.0, {"good_matches": 0, "inliers": 0, "kp_query": kp_query_count}
    
    inliers_ratio = total_inliers / max(1, total_good_matches)
    matches_ratio = total_good_matches / max(1, kp_query_count)
    
    score = 0.6 * inliers_ratio + 0.4 * matches_ratio
    
    debug_info = {
        "good_matches": total_good_matches,
        "inliers": total_inliers,
        "kp_query": kp_query_count
    }
    
    return min(score, 1.0), debug_info  # Normaliser à [0, 1]

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
            "best_class_12": None,
            "best_score": 0.0,
            "scores_per_class": {},
            "debug": {"error": "Impossible d'extraire les features ORB de la query"}
        }
    
    query_kp, query_desc = query_features
    kp_query_count = len(query_kp)
    
    # Déterminer les classes à tester
    classes_to_test = candidate_classes if candidate_classes else CLASSES_12
    
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
        "best_class_12": best_class,
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
    parser.add_argument('--n_refs', type=int, default=DEFAULT_N_REFS,
                       help=f'Nombre d\'images de référence par classe (défaut: {DEFAULT_N_REFS})')
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
        print(f"Meilleure classe: {result['best_class_12']}")
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

