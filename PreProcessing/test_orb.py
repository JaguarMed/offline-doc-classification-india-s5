import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from orb_matcher import load_orb_gallery, match_query_orb, ORB_CACHE_DIR
import argparse
from pathlib import Path
from PIL import Image
import io

try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    try:
        from pdf2image import convert_from_path
        HAS_PDF2IMAGE = True
    except ImportError:
        HAS_PDF2IMAGE = False

def extract_pdf_pages(pdf_path: str):
    pages = []
    try:
        if HAS_FITZ:
            pdf_doc = fitz.open(pdf_path)
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("ppm")
                pil_img = Image.open(io.BytesIO(img_data))
                pages.append((page_num + 1, pil_img))
            pdf_doc.close()
            return pages
        elif HAS_PDF2IMAGE:
            images = convert_from_path(pdf_path)
            return [(i + 1, img) for i, img in enumerate(images)]
        else:
            print("  Erreur: Aucune bibliotheque PDF disponible (PyMuPDF ou pdf2image)")
            return []
    except Exception as e:
        print(f"  Erreur lors de l'extraction du PDF: {e}")
        return []

def test_orb(query_path: str, dataset_root: str = r"C:\NLP-CV\dataset"):
    print(f"{'='*60}")
    print("TEST ORB SEUL")
    print(f"{'='*60}\n")
    
    print("Chargement de la galerie ORB...")
    orb_gallery = load_orb_gallery(ORB_CACHE_DIR)
    if orb_gallery is None:
        print("  Erreur: Galerie ORB non trouvee. Construisez-la d'abord avec:")
        print("  python orb_matcher.py --build_gallery --dataset_root \"{}\"".format(dataset_root))
        return
    
    print(f"  OK: {len(orb_gallery)} classes dans la galerie\n")
    
    print("Classes disponibles dans la galerie:")
    for class_name, ref_features in orb_gallery.items():
        num_refs = len(ref_features)
        if num_refs > 0:
            print(f"  - {class_name}: {num_refs} images de reference")
    print()
    
    query_path_lower = query_path.lower()
    is_pdf = query_path_lower.endswith('.pdf')
    
    if is_pdf:
        print(f"Traitement du PDF: {query_path}")
        pages = extract_pdf_pages(query_path)
        if not pages:
            print("  Erreur: Impossible d'extraire les pages du PDF")
            return
        
        print(f"  PDF contient {len(pages)} page(s)\n")
        
        all_page_results = []
        for page_num, pil_img in pages:
            print(f"{'='*60}")
            print(f"PAGE {page_num}")
            print(f"{'='*60}")
            
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                pil_img.save(tmp_file.name)
                tmp_path = tmp_file.name
            
            try:
                orb_result = match_query_orb(
                    tmp_path, 
                    orb_gallery, 
                    candidate_classes=None,
                    dataset_root=dataset_root
                )
                
                all_page_results.append({
                    'page': page_num,
                    'result': orb_result
                })
                
                print(f"\nMeilleure classe: {orb_result['best_class_4']}")
                print(f"Score: {orb_result['best_score']:.4f} ({orb_result['best_score']*100:.2f}%)")
                
                if orb_result['debug']:
                    print(f"\nDebug:")
                    for key, value in orb_result['debug'].items():
                        print(f"  {key}: {value}")
                
                print(f"\nScores par classe (top 5):")
                sorted_scores = sorted(orb_result['scores_per_class'].items(), key=lambda x: x[1], reverse=True)
                for i, (class_name, score) in enumerate(sorted_scores[:5], 1):
                    if score > 0:
                        print(f"  {i}. {class_name}: {score:.4f} ({score*100:.2f}%)")
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        print(f"\n{'='*60}")
        print("RESUME TOUTES LES PAGES")
        print(f"{'='*60}")
        for page_result in all_page_results:
            page_num = page_result['page']
            result = page_result['result']
            print(f"Page {page_num}: {result['best_class_4']} (score: {result['best_score']:.4f})")
        print(f"{'='*60}")
    else:
        print(f"Matching ORB pour: {query_path}")
        orb_result = match_query_orb(
            query_path, 
            orb_gallery, 
            candidate_classes=None,
            dataset_root=dataset_root
        )
        
        print(f"\n{'='*60}")
        print("RESULTATS ORB")
        print(f"{'='*60}")
        
        print(f"Meilleure classe: {orb_result['best_class_4']}")
        print(f"Score: {orb_result['best_score']:.4f} ({orb_result['best_score']*100:.2f}%)")
        
        if orb_result['debug']:
            print(f"\nDebug (meilleure classe):")
            for key, value in orb_result['debug'].items():
                print(f"  {key}: {value}")
        
        print(f"\nScores par classe (top 10):")
        sorted_scores = sorted(orb_result['scores_per_class'].items(), key=lambda x: x[1], reverse=True)
        for i, (class_name, score) in enumerate(sorted_scores[:10], 1):
            if score > 0:
                print(f"  {i}. {class_name}: {score:.4f} ({score*100:.2f}%)")
        
        print(f"{'='*60}")

def choose_file_with_explorer(initial_dir: str = None) -> str:
    if not HAS_TKINTER:
        print("Erreur: tkinter n'est pas disponible. Utilisez le mode interactif ou specifiez un fichier.")
        return None
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    if initial_dir is None:
        initial_dir = r"C:\NLP-CV\dataset"
    
    if not os.path.exists(initial_dir):
        initial_dir = os.getcwd()
    
    filetypes = [
        ("Tous les fichiers images et PDF", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp *.pdf"),
        ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
        ("PDF", "*.pdf"),
        ("Tous les fichiers", "*.*")
    ]
    
    file_path = filedialog.askopenfilename(
        title="Choisir un document ou une image pour la classification",
        initialdir=initial_dir,
        filetypes=filetypes
    )
    
    root.destroy()
    
    if file_path:
        print(f"\nFichier selectionne: {file_path}\n")
        return file_path
    else:
        print("\nAucun fichier selectionne.")
        return None

def main():
    parser = argparse.ArgumentParser(description='Test ORB seul pour classification de documents')
    parser.add_argument('query_image', type=str, nargs='?', default=None,
                       help='Chemin de l\'image query (optionnel)')
    parser.add_argument('--dataset_root', type=str, default=r"C:\NLP-CV\dataset",
                       help='Racine du dataset')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Ouvre l\'explorateur de fichiers pour choisir un fichier')
    
    args = parser.parse_args()
    
    if args.interactive or args.query_image is None:
        selected_file = choose_file_with_explorer(args.dataset_root)
        if selected_file is None:
            print("Aucun fichier selectionne.")
            return
        args.query_image = selected_file
    
    if not os.path.exists(args.query_image):
        print(f"Erreur: le fichier {args.query_image} n'existe pas")
        return
    
    test_orb(args.query_image, args.dataset_root)

if __name__ == "__main__":
    main()
