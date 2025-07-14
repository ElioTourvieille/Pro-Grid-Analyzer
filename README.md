# Pro Grid Analyzer 🃏

Application d'analyse de tournois de poker avec intelligence artificielle pour optimiser votre grind.

## ✨ Features

- **📊 Upload CSV Sharkscope** : Importez vos données de tournois directement depuis Sharkscope
- **🤖 Analyse IA** : Modèle ML qui prédit la probabilité de profitabilité de chaque tournoi
- **🎯 Recommandations personnalisées** : Top 5 des tournois les plus prometteurs selon votre profil
- **📈 Interface moderne** : Design dark/light mode avec Tailwind CSS
- **⚡ Performance** : Analyse de milliers de tournois en quelques secondes

## 🛠️ Tech Stack

### Frontend
- **Astro** - Framework web moderne
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling utilitaire
- **Shadcn/ui** - Composants UI

### Backend
- **FastAPI** - API Python moderne
- **Pandas** - Manipulation de données
- **XGBoost** - Modèle de machine learning
- **NumPy** - Calculs scientifiques

### ML/AI
- **Classification binaire** : Prédiction de profitabilité
- **Feature engineering** : Variables contextuelles (buy-in, participants, structure, etc.)
- **Cross-validation** : Évaluation robuste du modèle

## 🚀 Installation

### Prérequis
- Python 3.8+
- Node.js 16+
- pnpm (recommandé) ou npm

### Backend
```bash
cd fastapi-backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```
Le serveur démarre sur `http://localhost:8000`

### Frontend
```bash
cd astro-frontend
pnpm install
pnpm dev
```
L'application démarre sur `http://localhost:4321`

## 📖 Usage

1. **Upload de données** : Glissez-déposez votre fichier CSV Sharkscope
2. **Analyse automatique** : Le modèle analyse tous vos tournois
3. **Recommandations** : Consultez le top 5 des tournois les plus prometteurs
4. **Insights** : Découvrez les patterns qui marchent pour vous

## 📊 Format des données

L'application accepte les fichiers CSV Sharkscope avec les colonnes :
- `Nom du jeu` : Nom du tournoi
- `Mise` : Buy-in
- `Participants` : Nombre de joueurs
- `Rake` : Frais de la salle
- `Vitesse` : Structure (Normal, Turbo, etc.)
- `Structure` : Type de jeu
- `Résultat` : Performance du joueur

## 🎯 Fonctionnalités ML

### Modèle de classification
- **Cible** : `is_profitable` (tournoi profitable ou non)
- **Features** : Buy-in, participants, structure, rake, etc.
- **Métrique** : F1-score optimisé
- **Validation** : Cross-validation 5-fold

### Recommandations
- Top 5 des tournois avec la plus haute probabilité de profitabilité
- Regroupement automatique des doublons
- Raisons personnalisées par tournoi

## 🔧 Configuration

### Variables d'environnement
```bash
# Frontend (.env)
PUBLIC_API_URL=http://localhost:8000

# Backend (optionnel)
MODEL_PATH=app/model.pkl
```

### Modèle personnalisé
Pour utiliser votre propre modèle :
1. Entraînez votre modèle avec `train_model_v4.py`
2. Remplacez `app/model.pkl` dans le backend
3. Adaptez les features si nécessaire

## 📁 Structure du projet

```
Pro Grid Analyzer/
├── astro-frontend/          # Interface utilisateur
│   ├── src/
│   │   ├── components/      # Composants React/Astro
│   │   ├── pages/          # Pages de l'application
│   │   └── lib/            # Utilitaires et API client
│   └── package.json
├── fastapi-backend/         # API et ML
│   ├── app/
│   │   ├── main.py         # Endpoints FastAPI
│   │   ├── model_utils.py  # Utilitaires ML
│   │   └── model.pkl       # Modèle entraîné
│   └── requirements.txt
└── README.md
```

## 📝 License

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

**Pro Grid Analyzer** - Optimisez votre grind avec l'IA ! 🚀 