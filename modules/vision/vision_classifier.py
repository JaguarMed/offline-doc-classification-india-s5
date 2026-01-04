"""Classifieur d'images basé sur Vision Transformer (Swin/DeiT) pour CPU."""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import cv2


class VisionClassifier:
    """Classifieur d'images utilisant un Vision Transformer fine-tuné."""
    
    LABELS = ['CIN', 'releve_bancaire', 'facture_eau', 'facture_electricite', 'document_employeur']
    
    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "swin_tiny",
        image_size: int = 224,
        device: Optional[str] = None
    ):
        """
        Initialise le classifieur vision.
        
        Args:
            checkpoint_path: Chemin vers le checkpoint fine-tuné
            model_type: Type de modèle ("swin_tiny" ou "deit_small")
            image_size: Taille d'image (224 par défaut)
            device: Device ('cpu' ou 'cuda'), auto-détecté si None
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.model_type = model_type
        self.image_size = image_size
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Transformation d'image
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Charger le modèle
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Mapping labels
        self.id2label = {i: label for i, label in enumerate(self.LABELS)}
        self.label2id = {label: i for i, label in enumerate(self.LABELS)}
    
    def _load_model(self) -> nn.Module:
        """Charge le modèle depuis le checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Si le checkpoint contient 'model_state_dict', l'utiliser
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Créer le modèle de base selon le type
        if self.model_type == "swin_tiny":
            try:
                from torchvision.models import swin_t, Swin_T_Weights
                model = swin_t(weights=None)
            except ImportError:
                # Fallback pour versions plus anciennes
                from torchvision.models import swin_transformer
                model = swin_transformer.swin_t(weights=None)
            # Adapter la dernière couche pour 5 classes
            if hasattr(model, 'head'):
                model.head = nn.Linear(model.head.in_features, len(self.LABELS))
            elif hasattr(model, 'classifier'):
                model.classifier = nn.Linear(model.classifier.in_features, len(self.LABELS))
        elif self.model_type == "deit_small":
            try:
                from torchvision.models import deit_small_patch16_224, DeiT_Small_Weights
                model = deit_small_patch16_224(weights=None)
            except ImportError:
                # Fallback
                from torchvision.models import vision_transformer
                model = vision_transformer.vit_small_patch16_224(weights=None)
            if hasattr(model, 'head'):
                model.head = nn.Linear(model.head.in_features, len(self.LABELS))
            elif hasattr(model, 'classifier'):
                model.classifier = nn.Linear(model.classifier.in_features, len(self.LABELS))
        else:
            raise ValueError(f"Type de modèle non supporté: {self.model_type}")
        
        # Charger les poids
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Attention: Erreur lors du chargement des poids: {e}")
            print("Tentative de chargement partiel...")
        
        return model
    
    def preprocess_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> torch.Tensor:
        """Pré-traite une image pour l'inférence."""
        # Charger l'image
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                img = Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            img = image.convert('RGB')
        else:
            raise ValueError(f"Type d'image non supporté: {type(image)}")
        
        # Appliquer transformations
        tensor = self.transform(img).unsqueeze(0)  # Ajouter batch dimension
        return tensor
    
    def predict(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Dict:
        """
        Prédit la classe d'une image.
        
        Args:
            image: Image à classifier (chemin, array numpy, ou PIL Image)
        
        Returns:
            Dict avec 'label', 'confidence', 'probabilities'
        """
        # Pré-traitement
        tensor = self.preprocess_image(image)
        tensor = tensor.to(self.device)
        
        # Inférence
        with torch.inference_mode():
            outputs = self.model(tensor)
            logits = outputs
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
    
    def predict_batch(self, images: list) -> List[Dict]:
        """Prédit pour une liste d'images."""
        return [self.predict(img) for img in images]

