"""Permet de vérifier si un rapport PDF existe déjà dans un répertoire donné en utilisant fuzzy matching."""

import os
import re
from rapidfuzz import fuzz

def sanitize_company_name(name):
    """Nettoie le nom de l'entreprise : 'Thermo Fischer' -> 'thermo_fisher'"""
    name = name.lower()
    # Remplace tout ce qui n'est pas une lettre ou un chiffre par un underscore
    name = re.sub(r'[^a-z0-9]', '_', name)
    # Evite les doubles underscores
    name = re.sub(r'_+', '_', name)
    return name


def find_existing_reports(company, year, download_dir):
    """Vérifie si un ou plusieurs PDF existent déjà et retourne la liste des chemins."""
    if not os.path.exists(download_dir):
        return None

    target_name = sanitize_company_name(company)

    year_str = str(year)
    # 1. On initialise une liste pour stocker les rapports trouvés
    existing_reports = []
    
    for filename in os.listdir(download_dir):
        if not filename.endswith(".pdf"):
            continue
            
        file_base = filename.lower().replace(".pdf", "")
        
        # Vérification de l'année (critère éliminatoire)
        if year_str in file_base:
            # Préparation des chaînes pour la comparaison
            clean_filename = file_base.replace(year_str, "").replace("_", " ").strip()
            clean_target = target_name.replace("_", " ").strip()
            
            # Fuzzy match
            score = fuzz.partial_ratio(clean_target, clean_filename)
            
            if score >= 90:
                # 2. Au lieu de retourner tout de suite, on ajoute à la liste
                existing_reports.append(os.path.join(download_dir, filename))
    
    # 3. On retourne la liste si elle n'est pas vide, sinon None
    return existing_reports if existing_reports else None

