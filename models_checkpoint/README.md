# Models Checkpoints

Ce dossier contient les checkpoints des modèles fine-tunés pour la classification de documents marocains.

## Structure

```
models_checkpoint/
└── best_model_xlmr_large/
    ├── config.json              # Configuration du modèle
    ├── label_classes.npy        # Classes de labels
    ├── model.safetensors        # Poids du modèle (2.2 GB) - NON INCLUS
    ├── sentencepiece.bpe.model  # Tokenizer SentencePiece
    ├── special_tokens_map.json  # Tokens spéciaux
    ├── tokenizer.json           # Tokenizer
    ├── tokenizer_config.json    # Config tokenizer
    └── training_args.bin        # Arguments d'entraînement - NON INCLUS
```

## Téléchargement des poids

Les fichiers volumineux (`model.safetensors`, `training_args.bin`) ne sont pas inclus dans le dépôt Git.

### Option 1: Télécharger depuis HuggingFace Hub

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-large")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

# Puis fine-tuner sur vos données
```

### Option 2: Demander les poids

Contactez l'équipe du projet pour obtenir les poids fine-tunés.

### Option 3: Re-entraîner

Utilisez le notebook Colab fourni pour fine-tuner XLM-RoBERTa sur vos propres données.

## Modèle utilisé

- **Base**: `xlm-roberta-large` (XLM-RoBERTa Large)
- **Fine-tuning**: Classification de documents marocains (4 classes)
- **Classes**: CIN, RELEVE BANCAIRE, FACTURE, DOCUMENT EMPLOYEUR





