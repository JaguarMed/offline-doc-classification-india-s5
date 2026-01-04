"""
Helper functions pour convertir des PDFs en images
"""

import cv2
import numpy as np
from PIL import Image
import io

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

def pdf_to_images(pdf_path):
    """Convertit un PDF en liste d'images PIL"""
    images = []
    
    if HAS_FITZ:
        pdf_doc = fitz.open(pdf_path)
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom pour meilleure qualité
            img_data = pix.tobytes("ppm")
            pil_img = Image.open(io.BytesIO(img_data))
            images.append(pil_img)
        pdf_doc.close()
        return images
    elif HAS_PDF2IMAGE:
        images = convert_from_path(pdf_path)
        return images
    else:
        raise ImportError("Aucune bibliothèque PDF disponible (PyMuPDF ou pdf2image)")
    
    return images





