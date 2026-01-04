import pandas as pd
import os

excel_file = r"C:\NLP-CV\dataset\releve_bancaire_dataset.xlsx"

if os.path.exists(excel_file):
    df = pd.read_excel(excel_file)
    
    print(f"Fichier charge: {len(df)} lignes")
    print(f"Colonnes: {list(df.columns)}\n")
    
    if 'bank' in df.columns:
        print("Valeurs uniques AVANT correction:")
        print(df['bank'].value_counts().head(20))
        print()
        
        df['bank'] = df['bank'].str.replace('SOGETEL votre banque par tlphone', 'SOCIETE GENERALE', case=False, regex=False)
        df['bank'] = df['bank'].str.replace('SOGETEL votre banque par telephone', 'SOCIETE GENERALE', case=False, regex=False)
        df['bank'] = df['bank'].str.replace('OGETEL votre banque par telephone', 'SOCIETE GENERALE', case=False, regex=False)
        df['bank'] = df['bank'].str.replace('Directive de Bank Al-Maghrib toute', 'SOCIETE GENERALE', case=False, regex=False)
        df['bank'] = df['bank'].str.replace('Directive de Bank Al-Maghrib', 'SOCIETE GENERALE', case=False, regex=False)
        df['bank'] = df['bank'].str.replace('Banque Ville NCompte', 'CDM CREDIT DU MAROC', case=False, regex=False)
        df['bank'] = df['bank'].str.replace('SOGETEL', 'SOCIETE GENERALE', case=False, regex=False)
        df['bank'] = df['bank'].str.replace('OGETEL', 'SOCIETE GENERALE', case=False, regex=False)
        
        df.loc[df['bank'].str.contains('SOGETEL|OGETEL', case=False, na=False), 'bank'] = 'SOCIETE GENERALE'
        df.loc[df['bank'].str.contains('Directive de Bank', case=False, na=False), 'bank'] = 'SOCIETE GENERALE'
        df.loc[df['bank'].str.contains('votre banque par', case=False, na=False), 'bank'] = 'SOCIETE GENERALE'
        df.loc[df['bank'].str.contains('SOCIETE GENERALE toute', case=False, na=False), 'bank'] = 'SOCIETE GENERALE'
        df.loc[df['bank'].str.contains('SSOCIETE GENERALE', case=False, na=False), 'bank'] = 'SOCIETE GENERALE'
        
        if 'output' in df.columns:
            df['output'] = df['output'].str.replace('Releve bancaire SOGETEL votre banque par tlphone', 'Releve bancaire SOCIETE GENERALE', case=False, regex=False)
            df['output'] = df['output'].str.replace('Releve bancaire OGETEL votre banque par telephone', 'Releve bancaire SOCIETE GENERALE', case=False, regex=False)
            df['output'] = df['output'].str.replace('Releve bancaire Directive de Bank Al-Maghrib toute', 'Releve bancaire SOCIETE GENERALE', case=False, regex=False)
            df['output'] = df['output'].str.replace('Releve bancaire Directive de Bank Al-Maghrib', 'Releve bancaire SOCIETE GENERALE', case=False, regex=False)
            df['output'] = df['output'].str.replace('Releve bancaire Banque Ville NCompte', 'Releve bancaire CDM CREDIT DU MAROC', case=False, regex=False)
            df['output'] = df['output'].str.replace('Releve bancaire SOGETEL', 'Releve bancaire SOCIETE GENERALE', case=False, regex=False)
            df['output'] = df['output'].str.replace('Releve bancaire OGETEL', 'Releve bancaire SOCIETE GENERALE', case=False, regex=False)
            
            df.loc[df['output'].str.contains('SOGETEL|OGETEL', case=False, na=False), 'output'] = df.loc[df['output'].str.contains('SOGETEL|OGETEL', case=False, na=False), 'output'].str.replace('SOGETEL|OGETEL', 'SOCIETE GENERALE', case=False, regex=True)
            df.loc[df['output'].str.contains('Directive de Bank', case=False, na=False), 'output'] = df.loc[df['output'].str.contains('Directive de Bank', case=False, na=False), 'output'].str.replace('Directive de Bank.*', 'SOCIETE GENERALE', case=False, regex=True)
            df.loc[df['output'].str.contains('votre banque par', case=False, na=False), 'output'] = df.loc[df['output'].str.contains('votre banque par', case=False, na=False), 'output'].str.replace('.*votre banque par.*', 'Releve bancaire SOCIETE GENERALE', case=False, regex=True)
        
        print("Valeurs uniques APRES correction:")
        print(df['bank'].value_counts().head(20))
        print()
        
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"Fichier sauvegarde: {excel_file}")
    else:
        print("Colonne 'bank' introuvable")
else:
    print(f"Fichier introuvable: {excel_file}")

