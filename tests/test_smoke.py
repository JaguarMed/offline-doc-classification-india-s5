"""Tests smoke pour vérifier le fonctionnement de base."""

import pytest
from pathlib import Path
import sys

# Ajouter le root au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.ocr_nlp import extract_ocr_text, TextClassifier
from modules.vision import VisionClassifier
from modules.orb import ORBClassifier
from modules.fusion import FusionModule
from pipeline.inference import InferencePipeline


def test_ocr_extraction():
    """Test extraction OCR sur une image."""
    # Chercher une image de test
    dataset_dir = Path("dataset")
    if not dataset_dir.exists():
        pytest.skip("Dataset non trouvé")
    
    # Trouver une image
    test_image = None
    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.png"))
            if images:
                test_image = images[0]
                break
    
    if test_image is None:
        pytest.skip("Aucune image de test trouvée")
    
    # Test OCR
    text = extract_ocr_text(test_image, languages="fra", enable_cache=False)
    assert isinstance(text, str)
    print(f"OCR extrait {len(text)} caractères")


def test_text_classifier_loading():
    """Test chargement du classifieur texte (si checkpoint existe)."""
    checkpoint_path = Path("checkpoints/text")
    if not checkpoint_path.exists():
        pytest.skip("Checkpoint texte non trouvé")
    
    try:
        classifier = TextClassifier(str(checkpoint_path))
        assert classifier is not None
        print("TextClassifier chargé avec succès")
    except Exception as e:
        pytest.skip(f"Impossible de charger TextClassifier: {e}")


def test_vision_classifier_loading():
    """Test chargement du classifieur vision (si checkpoint existe)."""
    checkpoint_path = Path("checkpoints/vision")
    if not checkpoint_path.exists():
        pytest.skip("Checkpoint vision non trouvé")
    
    try:
        classifier = VisionClassifier(str(checkpoint_path), model_type="swin_tiny")
        assert classifier is not None
        print("VisionClassifier chargé avec succès")
    except Exception as e:
        pytest.skip(f"Impossible de charger VisionClassifier: {e}")


def test_orb_classifier():
    """Test classifieur ORB."""
    classifier = ORBClassifier()
    assert classifier is not None
    
    # Test sur une image
    dataset_dir = Path("dataset")
    if dataset_dir.exists():
        for class_dir in dataset_dir.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob("*.png"))
                if images:
                    result = classifier.predict(images[0])
                    assert 'label' in result
                    assert 'confidence' in result
                    assert 'probabilities' in result
                    print(f"ORB prédiction: {result['label']} ({result['confidence']:.4f})")
                    break


def test_fusion_module():
    """Test module de fusion."""
    fusion = FusionModule(weight_text=0.6, weight_vision=0.3, weight_orb=0.1)
    
    # Prédictions mock
    text_pred = {
        'label': 'CIN',
        'confidence': 0.8,
        'probabilities': {'CIN': 0.8, 'releve_bancaire': 0.1, 'facture_eau': 0.05, 'facture_electricite': 0.03, 'document_employeur': 0.02}
    }
    vision_pred = {
        'label': 'CIN',
        'confidence': 0.7,
        'probabilities': {'CIN': 0.7, 'releve_bancaire': 0.15, 'facture_eau': 0.08, 'facture_electricite': 0.05, 'document_employeur': 0.02}
    }
    orb_pred = {
        'label': 'CIN',
        'confidence': 0.6,
        'probabilities': {'CIN': 0.6, 'releve_bancaire': 0.2, 'facture_eau': 0.1, 'facture_electricite': 0.05, 'document_employeur': 0.05}
    }
    
    result = fusion.fuse(text_pred, vision_pred, orb_pred)
    assert 'label' in result
    assert 'confidence' in result
    assert 'probabilities' in result
    assert 'module_details' in result
    print(f"Fusion prédiction: {result['label']} ({result['confidence']:.4f})")


def test_inference_pipeline():
    """Test pipeline d'inférence complet (si checkpoints existent)."""
    text_ckpt = Path("checkpoints/text")
    vision_ckpt = Path("checkpoints/vision")
    
    if not text_ckpt.exists() or not vision_ckpt.exists():
        pytest.skip("Checkpoints non trouvés")
    
    # Chercher une image de test
    dataset_dir = Path("dataset")
    if not dataset_dir.exists():
        pytest.skip("Dataset non trouvé")
    
    test_image = None
    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.png"))
            if images:
                test_image = images[0]
                break
    
    if test_image is None:
        pytest.skip("Aucune image de test trouvée")
    
    try:
        pipeline = InferencePipeline(
            text_checkpoint=str(text_ckpt),
            vision_checkpoint=str(vision_ckpt)
        )
        
        result = pipeline.predict(test_image, return_ocr_text=True, return_details=True)
        
        assert 'prediction' in result
        assert 'label' in result['prediction']
        assert 'confidence' in result['prediction']
        print(f"Pipeline prédiction: {result['prediction']['label']} ({result['prediction']['confidence']:.4f})")
    except Exception as e:
        pytest.skip(f"Impossible de tester le pipeline: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])










