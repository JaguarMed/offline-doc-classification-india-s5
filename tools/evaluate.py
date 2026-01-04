"""Script pour évaluer un checkpoint et générer des artefacts d'analyse."""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ajouter le root au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.inference import InferencePipeline
import yaml


def get_git_hash():
    """Récupère le hash git du commit actuel si disponible."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None


def evaluate(
    dataset_dir: str,
    text_checkpoint: str,
    vision_checkpoint: str,
    output_dir: str,
    config_path: str = None,
    vision_model_type: str = "swin_tiny"
):
    """
    Évalue un checkpoint et génère les artefacts d'analyse.
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialiser le pipeline
    pipeline = InferencePipeline(
        config_path=config_path,
        text_checkpoint=text_checkpoint,
        vision_checkpoint=vision_checkpoint,
        vision_model_type=vision_model_type
    )
    
    # Charger config pour sauvegarder
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Parcourir le dataset
    predictions_data = []
    true_labels = []
    pred_labels = []
    
    # Parcourir les classes
    for class_dir in dataset_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        true_label = class_dir.name
        if true_label.startswith('.'):
            continue
        
        print(f"Évaluation classe: {true_label}")
        
        # Parcourir les images
        image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.pdf"))
        
        for img_path in tqdm(image_files, desc=f"  {true_label}"):
            try:
                # Prédiction
                result = pipeline.predict(img_path, return_ocr_text=True, return_details=True)
                
                pred_label = result['prediction']['label']
                confidence = result['prediction']['confidence']
                ocr_text = result.get('ocr_text', '')
                
                # Détails par module
                module_details = result['prediction'].get('module_details', {})
                text_conf = module_details.get('text', {}).get('confidence', 0.0)
                vision_conf = module_details.get('vision', {}).get('confidence', 0.0)
                orb_conf = module_details.get('orb', {}).get('confidence', 0.0)
                
                # Probabilités
                probs = result['prediction']['probabilities']
                probs_json = json.dumps(probs)
                
                predictions_data.append({
                    'file_path': str(img_path),
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': confidence,
                    'proba_json': probs_json,
                    'module_conf_text': text_conf,
                    'module_conf_vision': vision_conf,
                    'module_conf_orb': orb_conf,
                    'ocr_len': len(ocr_text),
                    'ocr_snippet': ocr_text[:200] + '...' if len(ocr_text) > 200 else ocr_text
                })
                
                true_labels.append(true_label)
                pred_labels.append(pred_label)
                
            except Exception as e:
                print(f"Erreur sur {img_path}: {e}")
                continue
    
    # Créer DataFrame
    df = pd.DataFrame(predictions_data)
    
    # Métriques
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, output_dict=True)
    
    # Confusion matrix
    labels = sorted(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    
    # Sauvegarder config
    config_save = {
        'text_checkpoint': text_checkpoint,
        'vision_checkpoint': vision_checkpoint,
        'vision_model_type': vision_model_type,
        'date': datetime.now().isoformat(),
        'git_hash': get_git_hash(),
        'fusion_weights': {
            'text': pipeline.fusion_module.weight_text,
            'vision': pipeline.fusion_module.weight_vision,
            'orb': pipeline.fusion_module.weight_orb
        }
    }
    
    with open(output_path / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config_save, f, indent=2, ensure_ascii=False)
    
    # Sauvegarder métriques
    metrics = {
        'accuracy': float(accuracy),
        'macro_f1': float(report['macro avg']['f1-score']),
        'weighted_f1': float(report['weighted avg']['f1-score']),
        'per_class': {}
    }
    
    for label in labels:
        if label in report:
            metrics['per_class'][label] = {
                'precision': float(report[label]['precision']),
                'recall': float(report[label]['recall']),
                'f1': float(report[label]['f1-score'])
            }
    
    with open(output_path / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # Sauvegarder prédictions
    df.to_csv(output_path / 'predictions.csv', index=False, encoding='utf-8')
    
    # Sauvegarder confusion matrix
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(output_path / 'confusion_matrix.csv')
    
    # Visualisation confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Dossier erreurs (optionnel)
    errors_dir = output_path / 'errors'
    errors_dir.mkdir(exist_ok=True)
    
    # Top erreurs
    errors = df[df['true_label'] != df['pred_label']].copy()
    errors = errors.sort_values('confidence', ascending=True)  # Plus basse confiance = plus intéressant
    
    if len(errors) > 0:
        errors.head(20).to_csv(errors_dir / 'top_errors.csv', index=False, encoding='utf-8')
    
    print(f"\nÉvaluation terminée!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"\nRésultats sauvegardés dans: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluer un checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Dossier dataset (structure par classe)")
    parser.add_argument("--text_ckpt", type=str, required=True, help="Chemin checkpoint texte")
    parser.add_argument("--vision_ckpt", type=str, required=True, help="Chemin checkpoint vision")
    parser.add_argument("--out", type=str, required=True, help="Dossier de sortie (runs/<run_id>)")
    parser.add_argument("--config", type=str, default=None, help="Chemin config.yaml (optionnel)")
    parser.add_argument("--vision_model", type=str, default="swin_tiny", choices=["swin_tiny", "deit_small"], help="Type modèle vision")
    
    args = parser.parse_args()
    evaluate(
        args.dataset,
        args.text_ckpt,
        args.vision_ckpt,
        args.out,
        args.config,
        args.vision_model
    )










