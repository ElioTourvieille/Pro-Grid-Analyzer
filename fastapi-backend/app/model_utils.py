import joblib
import pandas as pd
import numpy as np
import os

# ✅ Chargement unique du modèle v4
import os
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

# 🎯 Fonction principale de prédiction
def predict_with_model(profile, tournois):
    # Convertir les tournois en DataFrame
    df = pd.DataFrame(tournois)
    
    # Renommer les colonnes pour correspondre au modèle
    df = df.rename(columns={
        "Compétence_moyenne": "Compétence moyenne"
    })
    
    # Sauvegarder les noms pour le retour
    noms_tournois = df["Nom"].copy()
    
    # Supprimer la colonne Nom qui n'est pas une feature du modèle
    df = df.drop(columns=["Nom"])
    
    # Réorganiser les colonnes dans l'ordre exact du modèle d'entraînement
    colonnes_attendues = [
        "Participants", "Mise", "Rake", "Compétence", 
        "Compétence moyenne", "BuyIn_gap", "Est_turbo", "is_KO", 
        "Difficulte_relative", "Field_ratio", "Ratio_buyin_vs_ABI", 
        "ROI_sur_structure", "Heure_matin", "Heure_aprem", "Heure_soir", "Heure_nuit"
    ]
    
    # Debug: afficher les colonnes pour vérifier
    print("Colonnes avant prédiction:", df.columns.tolist())

    # Injecter les features du profil joueur (fixes)
    df["ABI"] = profile["ABI"]
    df["ROI_total"] = profile["ROI_total"]
    
    # Maintenant réorganiser
    df = df[colonnes_attendues]

    # Initialiser les colonnes horaires à 0
    for h in ["Heure_matin", "Heure_aprem", "Heure_soir", "Heure_nuit"]:
        df[h] = 0

    # Activer les bonnes heures selon le profil
    for h in profile["heures"]:
        if h == "matin":
            df["Heure_matin"] = 1
        elif h == "aprem":
            df["Heure_aprem"] = 1
        elif h == "soir":
            df["Heure_soir"] = 1
        elif h == "nuit":
            df["Heure_nuit"] = 1

    # ⚙️ Prédiction
    print("Colonnes finales:", df.columns.tolist())
    df["prediction"] = model.predict(df)

    # 🔍 Mock confiance et raisonnement pour affichage
    df["confiance"] = 0.75  # valeur fixe temporaire
    df["raisons"] = df.apply(lambda row: _get_mock_raisons(row), axis=1)
    
    # Remettre les noms des tournois
    df["Nom"] = noms_tournois

    # Grouper par Nom (ou Nom_id si tu veux être plus strict)
    grouped = df.groupby("Nom").agg({
        "Mise": "mean",
        "Rake": "mean",
        "Participants": "mean",
        "prediction": "max",  # ou "mean" si tu préfères
        "raisons": "first"  # ou une logique custom si tu veux
    }).reset_index()

    # Trie et prend les 5 meilleurs
    top_results = grouped.sort_values("prediction", ascending=False).head(5)

    results = top_results[["Nom", "Mise", "Rake", "Participants", "prediction", "raisons"]].to_dict(orient="records")

    return results

# --- FEATURE ENGINEERING ---
def create_base_features(df):
    df["is_profitable"] = (df["Résultat"] > 0).astype(int)
    df["Gain_net"] = df["Résultat"] - df["Rake"]
    df["Investissement_tournoi"] = df["Mise"] + df["Rake"]
    df["Est_turbo"] = df["Vitesse"].str.lower().str.contains("turbo").astype(int)
    tournoi_col = None
    for col in df.columns:
        if 'tournoi' in col.lower() and 'catalogue' in col.lower():
            tournoi_col = col
            break
    if tournoi_col:
        df["is_KO"] = df[tournoi_col].str.contains("KO|Bounty|PKO", case=False, na=False).astype(int)
    else:
        df["is_KO"] = df["Nom du jeu"].str.contains("KO|Bounty|PKO", case=False, na=False).astype(int)
    return df

def create_contextual_features(df):
    df["BuyIn_gap"] = df["Investissement_tournoi"] - df["ABI"]
    df["Difficulte_relative"] = df["Compétence"] - df["Compétence moyenne"]
    df["Field_ratio"] = df["Nb_jeux"] / df["Participants"]
    df["Ratio_buyin_vs_ABI"] = df["Investissement_tournoi"] / df["ABI"]
    return df

def create_performance_features(df):
    profit_par_structure = df.groupby(["Joueur", "Structure"])["Gain_net"].transform("sum")
    invest_par_structure = df.groupby(["Joueur", "Structure"])["Investissement_tournoi"].transform("sum")
    df["ROI_sur_structure"] = (profit_par_structure / invest_par_structure).replace([np.inf, -np.inf], np.nan)
    return df

def create_time_features(df):
    df["Heure"] = pd.to_datetime(df["Date :"], errors="coerce").dt.hour
    def horaire_bin(h):
        if pd.isna(h): return "inconnu"
        if 6 <= h < 12: return "matin"
        elif 12 <= h < 18: return "aprem"
        elif 18 <= h < 24: return "soir"
        else: return "nuit"
    df["Heure_bin"] = df["Heure"].apply(horaire_bin)
    df = pd.get_dummies(df, columns=['Heure_bin'], prefix='Heure')
    return df

# --- PIPELINE POUR L'API ---
def raw_csv_to_features(df_raw, abi, roi_total):
    df = df_raw.copy()
    df["ABI"] = abi
    df["ROI_total"] = roi_total
    df = create_base_features(df)
    df = create_contextual_features(df)
    df = create_performance_features(df)
    df = create_time_features(df)
    # Imputation des valeurs manquantes
    df['Compétence moyenne'].fillna(df['Compétence moyenne'].mean(), inplace=True)
    df['Difficulte_relative'].fillna(0, inplace=True)
    df['ROI_sur_structure'].fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    # Colonnes finales dans l'ordre du modèle
    colonnes_attendues = [
        'Participants', 'Mise', 'Rake', 'Compétence', 'ABI', 'ROI_total',
        'Compétence moyenne', 'BuyIn_gap', 'Est_turbo', 'is_KO',
        'Difficulte_relative', 'Field_ratio', 'Ratio_buyin_vs_ABI', 'ROI_sur_structure',
        'Heure_matin', 'Heure_aprem', 'Heure_soir', 'Heure_nuit'
    ]
    # Ajoute les colonnes horaires manquantes si besoin
    for h in ['Heure_matin', 'Heure_aprem', 'Heure_soir', 'Heure_nuit']:
        if h not in df.columns:
            df[h] = 0
    return df[colonnes_attendues]

# 🧠 Raisons fictives selon features
def _get_mock_raisons(row):
    raisons = []
    if row.get("is_KO") == 1:
        raisons.append("KO apprécié par le modèle")
    if row.get("Est_turbo") == 1:
        raisons.append("Structure turbo adaptée")
    if row.get("Participants", 0) < 500:
        raisons.append("Petit field recommandé")
    return raisons[:2] or ["Profil adapté"]
