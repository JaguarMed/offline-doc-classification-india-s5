"""Module OCR + NLP pour classification textuelle."""

from .ocr import extract_ocr_text, clean_text
from .text_classifier import TextClassifier

__all__ = ['extract_ocr_text', 'clean_text', 'TextClassifier']









