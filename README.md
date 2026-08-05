# ❤️ Heart Disease Prediction

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)

## 📌 Description
Ce projet propose un outil de **prédiction de maladie cardiaque** basé sur 11 caractéristiques cliniques (âge, pression artérielle, cholestérol, etc.). 
Plusieurs modèles de Machine Learning ont été entraînés et comparés. Une interface interactive **Streamlit** permet de tester les prédictions en temps réel.

## 📁 Structure du projet

heart_disease_prediction_project/

├── app/

│ └── app.py

├── data/

│ ├── raw/

│ │ └── heart.csv

│ └── processed/

│ └── heart_clean.csv

├── models/

│ ├── lr_final.pkl

│ ├── svm_final.pkl

│ ├── best_rf.pkl

│ ├── best_xgb.pkl

│ └── scaler.pkl

├── notebooks/

│ ├── DATA_MODELING.ipynb

│ ├── Data_PREPROCESSING.ipynb

│ └── DATA_VISUALISATION.ipynb

├── figures/

├── .gitignore

├── requirements.txt

└── README.md

## 📊 Données
- **Source** : [Heart Failure Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Volume** : 918 patients, 12 colonnes
- **Cible** : `HeartDisease` (1 = Malade, 0 = Sain)

## 🧠 Modèles testés
- Régression Logistique
- Decision Tree
- Random Forest
- XGBoost

### 🏆 Performances (Recall)
| Modèle | Recall |
|--------|-------------------|
| **XGBoost** | **~0.86** |
| **Decision Tree** | **~0.85** |
| **Random Forest** |**~0.83** |
| **Régression Logistique** | **~0.83** |

## 🔗 Tester l'application directement

👉 **[Cliquez ici pour accéder à l'application en ligne](https://heartdiseasepredictionproject.streamlit.app/)**
