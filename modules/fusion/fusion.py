"""Module de fusion soft-voting pondéré des 3 classifieurs."""

import numpy as np
from typing import Dict, List, Optional


class FusionModule:
    """Fusion des prédictions via soft voting pondéré."""
    
    LABELS = ['CIN', 'releve_bancaire', 'facture_eau', 'facture_electricite', 'document_employeur']
    
    def __init__(
        self,
        weight_text: float = 0.6,
        weight_vision: float = 0.3,
        weight_orb: float = 0.1
    ):
        """
        Initialise le module de fusion.
        
        Args:
            weight_text: Poids du module texte
            weight_vision: Poids du module vision
            weight_orb: Poids du module ORB
        """
        self.weight_text = weight_text
        self.weight_vision = weight_vision
        self.weight_orb = weight_orb
        
        # Normaliser les poids
        total = weight_text + weight_vision + weight_orb
        if total > 0:
            self.weight_text /= total
            self.weight_vision /= total
            self.weight_orb /= total
    
    def fuse(
        self,
        text_pred: Dict,
        vision_pred: Dict,
        orb_pred: Dict
    ) -> Dict:
        """
        Fusionne les prédictions des 3 modules.
        
        Args:
            text_pred: Prédiction du module texte {'label', 'confidence', 'probabilities'}
            vision_pred: Prédiction du module vision
            orb_pred: Prédiction du module ORB
        
        Returns:
            Dict avec 'label', 'confidence', 'probabilities', 'module_details'
        """
        # Extraire les probabilités de chaque module
        probs_text = np.array([text_pred['probabilities'].get(label, 0.0) for label in self.LABELS])
        probs_vision = np.array([vision_pred['probabilities'].get(label, 0.0) for label in self.LABELS])
        probs_orb = np.array([orb_pred['probabilities'].get(label, 0.0) for label in self.LABELS])
        
        # Fusion pondérée
        final_probs = (
            self.weight_text * probs_text +
            self.weight_vision * probs_vision +
            self.weight_orb * probs_orb
        )
        
        # Normaliser (au cas où)
        final_probs = final_probs / (final_probs.sum() + 1e-8)
        
        # Prédiction finale
        pred_id = int(np.argmax(final_probs))
        label = self.LABELS[pred_id]
        confidence = float(final_probs[pred_id])
        
        # Probabilités finales
        probabilities = {label: float(prob) for label, prob in zip(self.LABELS, final_probs)}
        
        # Détails par module
        module_details = {
            'text': {
                'label': text_pred['label'],
                'confidence': text_pred['confidence']
            },
            'vision': {
                'label': vision_pred['label'],
                'confidence': vision_pred['confidence']
            },
            'orb': {
                'label': orb_pred['label'],
                'confidence': orb_pred['confidence']
            }
        }
        
        return {
            'label': label,
            'confidence': confidence,
            'probabilities': probabilities,
            'module_details': module_details
        }









