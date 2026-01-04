import pandas as pd
import os

input_file = r"C:\NLP-CV\dataset\Data.xlsx"
output_file = r"C:\NLP-CV\dataset\Data.xlsx"

if os.path.exists(input_file):
    df = pd.read_excel(input_file)
    df = df.reset_index(drop=True)
    
    print(f"Fichier charge: {len(df)} lignes")
    print(f"Colonnes: {list(df.columns)}\n")
    
    def get_main_category(output_precision):
        output_str = str(output_precision).upper()
        
        if 'CIN' in output_str or 'CARTE' in output_str or 'IDENTITE' in output_str:
            return 'CIN'
        elif 'RELEVE BANCAIRE' in output_str or 'RELEVE BANQUAIRE' in output_str:
            return 'RELEVE BANCAIRE'
        elif 'FACTURE' in output_str:
            return 'FACTURE D\'EAU ET D\'ELECTRICITE'
        elif 'CONTRAT' in output_str or 'FICHE DE PAIE' in output_str or 'BULLETIN DE SALAIRE' in output_str or 'ATTESTATION' in output_str or 'CERTIFICAT' in output_str or 'AVIS' in output_str or 'LETTRE' in output_str:
            return 'DOCUMENT EMPLOYEUR'
        else:
            return 'AUTRE'
    
    if 'output_precision' in df.columns and 'output' in df.columns:
        print("Les colonnes output et output_precision existent deja. Mise a jour de output...")
        df['output'] = df['output_precision'].apply(get_main_category)
        df = df[['input', 'output', 'output_precision']]
        
    elif 'output' in df.columns:
        df = df.rename(columns={'output': 'output_precision'})
        df['output'] = df['output_precision'].apply(get_main_category)
        df = df[['input', 'output', 'output_precision']]
    else:
        print("Colonne 'output' introuvable")
        exit()
    
    print("Valeurs uniques dans output_precision:")
    print(df['output_precision'].value_counts())
    print("\n" + "="*60 + "\n")
    print("Valeurs uniques dans output (4 grandes classes):")
    print(df['output'].value_counts())
    print("\n" + "="*60 + "\n")
    print("Distribution par classe principale:")
    for category in df['output'].unique():
        print(f"\n{category}:")
        print(df[df['output'] == category]['output_precision'].value_counts().head(10))
    
    df.to_excel(output_file, index=False, engine='openpyxl')
    
    print(f"\n\nFichier sauvegarde: {output_file}")
    print(f"Colonnes finales: {list(df.columns)}")
    print(f"Nombre de lignes: {len(df)}")
else:
    print(f"Fichier introuvable: {input_file}")
