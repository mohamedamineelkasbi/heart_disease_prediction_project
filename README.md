# ❤️ Heart Disease Prediction

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)

## 📌 Description
Ce projet propose un outil de **prédiction de maladie cardiaque** basé sur 11 caractéristiques cliniques (âge, pression artérielle, cholestérol, etc.). 
Plusieurs modèles de Machine Learning ont été entraînés et comparés. Une interface interactive **Streamlit** permet de tester les prédictions en temps réel.

## 📁 Structure du projet

heart_disease_prediction_project/
│
├── app/ # 🖥️ Interface utilisateur (Streamlit)
│ └── app.py # Code principal de l'application
│
├── data/ # 📊 Données
│ ├── raw/ # Données brutes (non modifiées)
│ │ └── heart.csv # Dataset Kaggle (918 patients)
│ └── processed/ # Données après transformation
│ └── heart_clean.csv # Données imputées et encodées
│
├── models/ # 🤖 Modèles entraînés
│ ├── lr_final.pkl # Régression Logistique
│ ├── svm_final.pkl # Support Vector Machine
│ ├── best_rf.pkl # Random Forest
│ ├── best_xgb.pkl # XGBoost 
│ └── scaler.pkl # StandardScaler pour la normalisation
│
├── notebooks/ # 📓 Notebooks Jupyter
│ └── DATA_MODELING.ipynb # Pipeline complet (EDA → Modélisation)
│
├── reports/ # 📈 Visualisations et figures
│ └── (graphiques générés)
│
├── .gitignore # 🙈 Fichiers exclus du versionnement
├── requirements.txt # 📦 Dépendances Python
├── README.md # 📝 Documentation
└── start.bat # 🚀 Script de lancement

## 📊 Données
- **Source** : [Heart Failure Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Volume** : 918 patients, 12 colonnes
- **Cible** : `HeartDisease` (1 = Malade, 0 = Sain)

## 🧠 Modèles testés
- Régression Logistique
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

### 🏆 Performances (Recall)
| Modèle | Recall (seuil 0.5) |
|--------|-------------------|
| **XGBoost** (Meilleur) | **~0.88** |
| Régression Logistique | ~0.87 |
| Random Forest | ~0.81 |
| SVM | ~0.84 |

## 🛠️ Installation et Lancement

```bash
# 1. Cloner le dépôt
git clone https://github.com/ton-utilisateur/heart_disease_prediction_project.git
cd heart_disease_prediction_project

# 2. Créer un environnement virtuel (recommandé)
python -m venv .venv
# .venv\Scripts\activate   (Windows)
# source .venv/bin/activate (Mac/Linux)

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app/app.py