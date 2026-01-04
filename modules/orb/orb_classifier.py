"""Classifieur basé sur ORB (Oriented FAST and Rotated BRIEF) pour motifs visuels."""

import cv2
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path


class ORBClassifier:
    """Classifieur heuristique basé sur ORB pour détecter des motifs caractéristiques."""
    
    LABELS = ['CIN', 'releve_bancaire', 'facture_eau', 'facture_electricite', 'document_employeur']
    
    def __init__(
        self,
        max_features: int = 500,
        scale_factor: float = 1.2,
        n_levels: int = 8
    ):
        """
        Initialise le classifieur ORB.
        
        Args:
            max_features: Nombre max de features ORB
            scale_factor: Facteur d'échelle pour la pyramide
            n_levels: Nombre de niveaux dans la pyramide
        """
        self.max_features = max_features
        self.orb = cv2.ORB_create(nfeatures=max_features, scaleFactor=scale_factor, nlevels=n_levels)
        
        # Templates de référence (optionnel, peut être chargé depuis fichiers)
        self.templates = {}  # {label: [keypoints, descriptors]}
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Charge une image depuis un fichier."""
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        if image_path.suffix.lower() == '.pdf':
            images = convert_from_path(str(image_path), first_page=1, last_page=1)
            if not images:
                raise ValueError(f"Impossible de charger le PDF: {image_path}")
            img_array = np.array(images[0])
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return img_array
        else:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Impossible de charger l'image: {image_path}")
            return img
    
    def extract_features(self, image: np.ndarray) -> tuple:
        """Extrait les features ORB d'une image."""
        # Convertir en niveaux de gris si nécessaire
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Détecter keypoints et descripteurs
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def compute_heuristic_scores(self, image: np.ndarray) -> Dict[str, float]:
        """
        Calcule des scores heuristiques basés sur des caractéristiques visuelles.
        
        Cette méthode est simplifiée et peut être améliorée avec des templates.
        """
        # Convertir en niveaux de gris
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Extraire features
        keypoints, descriptors = self.extract_features(image)
        n_features = len(keypoints) if keypoints else 0
        
        # Caractéristiques de l'image
        h, w = gray.shape
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Densité de features
        feature_density = n_features / (h * w) if (h * w) > 0 else 0
        
        # Détection de lignes (pour tableaux)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        n_lines = len(lines) if lines is not None else 0
        
        # Scores heuristiques par classe (simplifié)
        scores = {}
        
        # CIN: généralement portrait, ratio proche de 1, beaucoup de texte
        cin_score = 0.3 if 0.6 < aspect_ratio < 1.4 else 0.1
        cin_score += 0.2 if n_features > 200 else 0.0
        
        # Relevé bancaire: souvent paysage, beaucoup de lignes (tableau)
        releve_score = 0.3 if aspect_ratio > 1.5 else 0.1
        releve_score += 0.3 if n_lines > 20 else 0.0
        
        # Factures: souvent paysage, code-barres potentiel
        facture_score = 0.3 if aspect_ratio > 1.3 else 0.1
        facture_score += 0.2 if feature_density > 0.001 else 0.0
        
        scores['CIN'] = max(0.1, min(0.9, cin_score))
        scores['releve_bancaire'] = max(0.1, min(0.9, releve_score))
        scores['facture_eau'] = max(0.1, min(0.9, facture_score * 0.9))
        scores['facture_electricite'] = max(0.1, min(0.9, facture_score * 0.95))
        scores['document_employeur'] = max(0.1, min(0.9, 0.5))  # Neutre par défaut
        
        # Normaliser en probas
        total = sum(scores.values())
        probabilities = {label: score / total for label, score in scores.items()}
        
        return probabilities
    
    def predict(
        self,
        image: Union[str, Path, np.ndarray]
    ) -> Dict:
        """
        Prédit la classe d'une image via ORB.
        
        Args:
            image: Image à classifier
        
        Returns:
            Dict avec 'label', 'confidence', 'probabilities', 'metadata'
        """
        # Charger l'image si nécessaire
        if isinstance(image, (str, Path)):
            img_array = self.load_image(image)
        else:
            img_array = image.copy()
        
        # Calculer scores heuristiques
        probabilities = self.compute_heuristic_scores(img_array)
        
        # Extraire features pour metadata
        keypoints, descriptors = self.extract_features(img_array)
        n_features = len(keypoints) if keypoints else 0
        
        # Prédiction
        label = max(probabilities, key=probabilities.get)
        confidence = probabilities[label]
        
        return {
            'label': label,
            'confidence': confidence,
            'probabilities': probabilities,
            'metadata': {
                'n_keypoints': n_features,
                'n_descriptors': n_features
            }
        }
    
    def predict_batch(self, images: list) -> List[Dict]:
        """Prédit pour une liste d'images."""
        return [self.predict(img) for img in images]









