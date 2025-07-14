from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import numpy as np

from model_utils import model, _get_mock_raisons
import pandas as pd

app = FastAPI(title="Pro Grid Analyzer API", version="1.0.0")

# CORS pour l'astro-frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "http://127.0.0.1:4321"],  # Astro dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles Pydantic
class PredictionResponse(BaseModel):
    Nom: str
    prediction: float
    confiance: float
    raisons: List[str]

@app.get("/")
async def root():
    return {"message": "Pro Grid Analyzer API"}



@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": True}

@app.get("/api/test")
async def test_endpoint():
    return {"message": "API is working"}

@app.post("/api/upload")
async def upload_tournaments(file: UploadFile = File(...)):
    try:
        # Lire le contenu du fichier
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Parser le CSV
        import csv
        from io import StringIO
        
        tournaments = []
        csv_reader = csv.DictReader(StringIO(content_str))
        
        for row in csv_reader:
            # Mapper les colonnes CSV vers notre format
            tournament = {
                "Nom": row.get("Nom", row.get("name", "Tournoi")),
                "Mise": float(row.get("Mise", row.get("buyIn", 0))),
                "Participants": int(row.get("Participants", row.get("players", 0))),
                "Date :": row.get("Date", row.get("startTime", "")),
                "Rake": float(row.get("Rake", 0)),
                "Compétence": float(row.get("Compétence", 70)),
                "Compétence moyenne": float(row.get("Compétence moyenne", 65)),
                "Nb_jeux": int(row.get("Nb_jeux", 50)),
                "Type": row.get("Type", row.get("type", "mtt")),
                "Prix": float(row.get("Prix", row.get("prizePool", 0)))
            }
            tournaments.append(tournament)
        
        return tournaments
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur upload: {str(e)}")

@app.post("/api/analyze")
async def analyze_file(file: UploadFile = File(...)):
    try:
        print(f"Fichier reçu: {file.filename}")
        
        # Parser le CSV
        content = await file.read()
        content_str = content.decode('utf-8')
        print(f"Contenu CSV: {content_str[:200]}...")
        
        import csv
        from io import StringIO
        
        tournaments = []
        csv_reader = csv.DictReader(StringIO(content_str))
        
        for row in csv_reader:
            # Nettoie les espaces autour des clés
            row = {k.strip(): v for k, v in row.items()}
            print(f"Ligne CSV: {row}")
            tournament = {
                "Nom": row.get("Nom", "Tournoi"),
                "Nom_id": row.get("Nom du jeu", ""), 
                "Mise": safe_float(row.get("Mise", 0)),
                "Participants": safe_int(row.get("Participants", 0)),
                "Date :": row.get("Date :", ""),
                "Rake": safe_float(row.get("Rake", 0)),
                "Vitesse": row.get("Vitesse", "Normal"),
                "Structure": row.get("Structure", "standard"),
                "Prix": safe_float(row.get("Prix", 0)),
                "Résultat": safe_float(row.get("Résultat", 0)),
                "Joueur": row.get("Joueur", "user"),
                "Compétence": 70,
                "Compétence moyenne": 65,
                "Nb_jeux": 50,
                "Type": "mtt"
            }
            tournaments.append(tournament)
        
        # Créer DataFrame et ajouter colonnes manquantes
        df = pd.DataFrame(tournaments)
        print(f"DataFrame créé avec {len(df)} lignes et colonnes: {df.columns.tolist()}")
        
        if "Compétence" not in df.columns:
            df["Compétence"] = 70
        if "Compétence moyenne" not in df.columns:
            df["Compétence moyenne"] = 65
        if "Rake" not in df.columns:
            df["Rake"] = 0
        if "Nb_jeux" not in df.columns:
            df["Nb_jeux"] = 50
        if "Vitesse" not in df.columns:
            df["Vitesse"] = "normal"
        if "Résultat" not in df.columns:
            df["Résultat"] = 0
        if "Joueur" not in df.columns:
            df["Joueur"] = "user"
        if "Structure" not in df.columns:
            df["Structure"] = "standard"
        
        # Utiliser des valeurs par défaut pour ABI et ROI
        abi_default = 50
        roi_default = 15
        
        # Créer des features simples pour le modèle
        df["ABI"] = abi_default
        df["ROI_total"] = roi_default
        
        # Ajouter les colonnes manquantes avec des valeurs par défaut
        required_columns = [
            'Participants', 'Mise', 'Rake', 'Compétence', 'ABI', 'ROI_total',
            'Compétence moyenne', 'BuyIn_gap', 'Est_turbo', 'is_KO',
            'Difficulte_relative', 'Field_ratio', 'Ratio_buyin_vs_ABI', 'ROI_sur_structure',
            'Heure_matin', 'Heure_aprem', 'Heure_soir', 'Heure_nuit'
        ]
        
        # Calculer les features manquantes avec gestion des erreurs
        df["BuyIn_gap"] = df["Mise"] - df["ABI"]
        df["Difficulte_relative"] = df["Compétence"] - df["Compétence moyenne"]
        
        # Éviter division par zéro
        df["Field_ratio"] = df["Nb_jeux"] / df["Participants"].replace(0, 1)
        df["Ratio_buyin_vs_ABI"] = df["Mise"] / df["ABI"].replace(0, 1)
        
        # Remplacer inf et -inf par des valeurs finies
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        df["ROI_sur_structure"] = 0  # Valeur par défaut
        df["Est_turbo"] = 0  # Valeur par défaut
        df["is_KO"] = 0  # Valeur par défaut
        
        # Colonnes horaires
        df["Heure_matin"] = 0
        df["Heure_aprem"] = 0
        df["Heure_soir"] = 0
        df["Heure_nuit"] = 0
        
        # S'assurer que toutes les colonnes sont présentes
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Prédiction
        print(f"Colonnes finales: {df.columns.tolist()}")
        features_df = df[required_columns]
        print(f"Features pour modèle: {features_df.shape}")
        print(f"Types de données: {features_df.dtypes}")
        
        # Vérifier et nettoyer les données avant prédiction
        print(f"Valeurs min/max: {features_df.min()}")
        print(f"Valeurs max: {features_df.max()}")
        
        # Convertir en float32 et remplacer les valeurs problématiques
        features_df = features_df.astype('float32')
        features_df = features_df.replace([np.inf, -np.inf], 0)
        features_df = features_df.fillna(0)
        
        try:
            print("Début de la prédiction...")
            # Utilise predict_proba pour la proba d'être profitable
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(features_df)[:, 1]
            else:
                probas = model.predict(features_df)  # fallback
            features_df["proba_profitable"] = probas

            # Raison mock: feature la plus importante (optionnel)
            importances = getattr(model, 'feature_importances_', None)
            if importances is not None:
                main_feature = features_df.columns[np.argmax(importances)]
            else:
                main_feature = "Profil global"

            features_df["raison"] = features_df.apply(get_reason, axis=1)

            # Après avoir calculé proba_profitable et raison
            features_df["Nom"] = df["Nom"]
            features_df["Mise"] = df["Mise"]
            features_df["Rake"] = df["Rake"]
            features_df["Participants"] = df["Participants"]

            # Grouper par Nom pour éviter les doublons
            grouped = features_df.groupby("Nom").agg({
                "Mise": "mean",
                "Rake": "mean", 
                "Participants": lambda x: round(x.mean()),
                "proba_profitable": "max",
                "raison": "first"
            }).reset_index()

            # Trie et prend les 5 meilleurs
            top_results = grouped.sort_values("proba_profitable", ascending=False).head(5)

            results = top_results[["Nom", "Mise", "Rake", "Participants", "proba_profitable", "raison"]].to_dict(orient="records")

            summary = {
                "nb_recommandees": int((features_df["proba_profitable"] > 0.5).sum()),
                "max_proba": float(features_df["proba_profitable"].max()),
                "min_proba": float(features_df["proba_profitable"].min()),
                "total_tournois": int(len(features_df))
            }
            return {"results": results, "summary": summary}
            
        except Exception as e:
            print(f"Erreur lors de la prédiction: {str(e)}")
            raise e
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur analyse: {str(e)}")

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

def safe_int(val):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0

def get_reason(row):
    reasons = []
    if row['Participants'] > 10000:
        reasons.append("Gros field, variance élevée")
    if row['Mise'] > 50:
        reasons.append("Buy-in élevé, attention à la gestion de bankroll")
    if row['Est_turbo'] == 1:
        reasons.append("Structure turbo, favorise l’agressivité")
    if row['is_KO'] == 1:
        reasons.append("Format KO, adapté à ton style")
    if not reasons:
        reasons.append("Profil global adapté")
    return ', '.join(reasons)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
