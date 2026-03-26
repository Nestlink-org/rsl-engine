import fitz  # PyMuPDF
from pdf2image import convert_from_path
from typing import List
import numpy as np
import cv2
from PIL import Image
import os

def pdf_to_images(pdf_path: str, dpi: int = 200) -> List[np.ndarray]:
    """Convert PDF to list of numpy arrays (images)"""
    try:
        # Try using PyMuPDF first (faster)
        images = []
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(np.array(img))
        
        pdf_document.close()
        return images
    
    except Exception as e:
        # Fallback to pdf2image
        images = convert_from_path(pdf_path, dpi=dpi)
        return [np.array(img) for img in images]

def get_pdf_page_count(pdf_path: str) -> int:
    """Get PDF page count"""
    doc = fitz.open(pdf_path)
    count = doc.page_count
    doc.close()
    return count