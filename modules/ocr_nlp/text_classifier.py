"""Classifieur de texte basé sur mDeBERTa fine-tuné."""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path


class TextClassifier:
    """Classifieur de texte utilisant un modèle Transformer fine-tuné."""
    
    LABELS = ['CIN', 'releve_bancaire', 'facture_eau', 'facture_electricite', 'document_employeur']
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialise le classifieur texte.
        
        Args:
            checkpoint_path: Chemin vers le checkpoint mDeBERTa fine-tuné
            device: Device ('cpu' ou 'cuda'), auto-détecté si None
            max_length: Longueur max des séquences
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.max_length = max_length
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Charger tokenizer et modèle
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.checkpoint_path),
            num_labels=len(self.LABELS)
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Mapping label_id -> label_name
        self.id2label = {i: label for i, label in enumerate(self.LABELS)}
        self.label2id = {label: i for i, label in enumerate(self.LABELS)}
    
    def predict(self, text: str) -> Dict:
        """
        Prédit la classe d'un texte.
        
        Args:
            text: Texte à classifier
        
        Returns:
            Dict avec 'label', 'confidence', 'probabilities'
        """
        if not text or not text.strip():
            # Texte vide: probas uniformes
            probs = np.ones(len(self.LABELS)) / len(self.LABELS)
            return {
                'label': self.LABELS[0],
                'confidence': float(probs[0]),
                'probabilities': {label: float(prob) for label, prob in zip(self.LABELS, probs)}
            }
        
        # Tokenisation
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inférence
        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Résultat
        pred_id = int(np.argmax(probs))
        label = self.id2label[pred_id]
        confidence = float(probs[pred_id])
        
        probabilities = {self.id2label[i]: float(probs[i]) for i in range(len(self.LABELS))}
        
        return {
            'label': label,
            'confidence': confidence,
            'probabilities': probabilities
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Prédit pour une liste de textes."""
        return [self.predict(text) for text in texts]







