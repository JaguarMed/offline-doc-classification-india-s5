"""Module OCR avec cache pour extraction de texte depuis images/PDFs."""

import os
import hashlib
import json
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path


def image_hash(image: np.ndarray) -> str:
    """Calcule un hash MD5 de l'image pour le cache."""
    return hashlib.md5(image.tobytes()).hexdigest()


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Charge une image depuis un fichier."""
    if isinstance(image_path, str):
        image_path = Path(image_path)
    
    if image_path.suffix.lower() == '.pdf':
        # Convertir première page PDF en image
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


def preprocess_image(image: np.ndarray, grayscale: bool = True, threshold: bool = False) -> np.ndarray:
    """Pré-traitement léger de l'image pour OCR."""
    if grayscale:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if threshold:
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return image


def clean_text(text: str) -> str:
    """Nettoie le texte OCR: normalise espaces, enlève répétitions de lignes vides."""
    if not text:
        return ""
    
    # Normaliser les espaces multiples
    lines = text.split('\n')
    cleaned_lines = []
    prev_empty = False
    
    for line in lines:
        # Normaliser espaces dans la ligne
        line = ' '.join(line.split())
        
        # Éviter répétitions de lignes vides
        if line.strip():
            cleaned_lines.append(line)
            prev_empty = False
        elif not prev_empty:
            cleaned_lines.append("")
            prev_empty = True
    
    result = '\n'.join(cleaned_lines).strip()
    return result


def extract_ocr_text(
    image_path: Union[str, Path, np.ndarray],
    languages: str = "fra+ara",
    cache_dir: Optional[Union[str, Path]] = None,
    enable_cache: bool = True,
    grayscale: bool = True,
    threshold: bool = False
) -> str:
    """
    Extrait le texte d'une image/PDF via OCR avec cache.
    
    Args:
        image_path: Chemin vers l'image/PDF ou array numpy
        languages: Langues OCR (ex: "fra+ara", "fra")
        cache_dir: Dossier pour le cache OCR
        enable_cache: Activer le cache
        grayscale: Convertir en niveaux de gris
        threshold: Appliquer seuillage
    
    Returns:
        Texte extrait nettoyé
    """
    # Charger l'image si c'est un chemin
    if isinstance(image_path, (str, Path)):
        image = load_image(image_path)
    else:
        image = image_path.copy()
    
    # Pré-traitement
    processed_image = preprocess_image(image, grayscale=grayscale, threshold=threshold)
    
    # Cache
    if enable_cache and cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        img_hash = image_hash(processed_image)
        cache_file = cache_dir / f"{img_hash}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                if cached_data.get('languages') == languages:
                    return cached_data.get('text', '')
    
    # OCR
    try:
        text = pytesseract.image_to_string(processed_image, lang=languages)
    except Exception as e:
        print(f"Erreur OCR: {e}")
        text = ""
    
    # Nettoyage
    cleaned = clean_text(text)
    
    # Sauvegarder dans le cache
    if enable_cache and cache_dir:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({'languages': languages, 'text': cleaned}, f, ensure_ascii=False)
    
    return cleaned











