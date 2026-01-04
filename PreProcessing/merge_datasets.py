import pandas as pd
import os

datasets = [
    {
        'file': r"C:\NLP-CV\dataset\releve_bancaire_dataset.xlsx",
        'exists': False
    },
    {
        'file': r"C:\NLP-CV\dataset\facture_dataset.xlsx",
        'exists': False
    },
    {
        'file': r"C:\NLP-CV\dataset\document_admin_dataset.xlsx",
        'exists': False
    },
    {
        'file': r"C:\NLP-CV\dataset\cin_front_back_2cols.xlsx",
        'exists': False,
        'columns': ['input', 'output']
    }
]

output_file = r"C:\NLP-CV\dataset\Data.xlsx"

all_data = []

print("Verification des fichiers...\n")

for ds in datasets:
    if os.path.exists(ds['file']):
        ds['exists'] = True
        print(f"[OK] Trouve: {os.path.basename(ds['file'])}")
        try:
            df = pd.read_excel(ds['file'])
            if 'input' in df.columns and 'output' in df.columns:
                df_filtered = df[['input', 'output']].copy()
                # Ne pas normaliser CIN_front et CIN_back - préserver les classes exactes
                # if ds['file'].endswith('cin_front_back_2cols.xlsx'):
                #     df_filtered['output'] = 'CIN'
                all_data.append(df_filtered)
                print(f"  - {len(df_filtered)} lignes chargees")
            else:
                print(f"  - Colonnes input/output manquantes")
                print(f"  - Colonnes disponibles: {list(df.columns)}")
        except Exception as e:
            print(f"  - Erreur: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[X] Non trouve: {os.path.basename(ds['file'])}")

if len(all_data) > 0:
    print(f"\nFusion des {len(all_data)} datasets...")
    merged_df = pd.concat(all_data, ignore_index=True)
    
    print(f"Nombre total de lignes avant filtrage: {len(merged_df)}")
    
    merged_df['input'] = merged_df['input'].astype(str)
    merged_df['output'] = merged_df['output'].astype(str)
    
    merged_df = merged_df[merged_df['input'].notna()]
    merged_df = merged_df[merged_df['output'].notna()]
    
    merged_df = merged_df[merged_df['input'] != 'nan']
    merged_df = merged_df[merged_df['output'] != 'nan']
    
    merged_df = merged_df[merged_df['input'] != '']
    merged_df = merged_df[merged_df['output'] != '']
    
    merged_df['input'] = merged_df['input'].str.strip()
    merged_df['output'] = merged_df['output'].str.strip()
    
    merged_df = merged_df[merged_df['input'] != '']
    merged_df = merged_df[merged_df['output'] != '']
    
    merged_df = merged_df[merged_df['output'] != 'Non identifiée']
    
    merged_df = merged_df[merged_df['input'].str.len() >= 10]
    merged_df = merged_df[merged_df['output'].str.len() >= 3]
    
    merged_df['output'] = merged_df['output'].str.replace('SOCIETE GENERALE toute', 'SOCIETE GENERALE', case=False, regex=False)
    merged_df['output'] = merged_df['output'].str.replace('Releve bancaire SOCIETE GENERALE toute', 'Releve bancaire SOCIETE GENERALE', case=False, regex=False)
    
    merged_df['output'] = merged_df['output'].str.replace('Releve bancaire BANK CIHD', 'Releve bancaire CIH BANK', case=False, regex=False)
    merged_df['output'] = merged_df['output'].str.replace('Releve bancaire BANK CIH', 'Releve bancaire CIH BANK', case=False, regex=False)
    merged_df['output'] = merged_df['output'].str.replace('Releve bancaire CIH BANK CIH', 'Releve bancaire CIH BANK', case=False, regex=False)
    
    merged_df = merged_df.drop_duplicates(subset=['input'], keep='first')
    
    merged_df = merged_df.reset_index(drop=True)
    
    print(f"Nombre total de lignes apres filtrage: {len(merged_df)}")
    
    print(f"\nValeurs uniques dans output:")
    print(merged_df['output'].value_counts())
    
    merged_df.to_excel(output_file, index=False, engine='openpyxl')
    
    print(f"\nDataset fusionne sauvegarde: {output_file}")
    print(f"Colonnes: {list(merged_df.columns)}")
else:
    print("\nAucun dataset trouve a fusionner.")

