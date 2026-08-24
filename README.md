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

│    ├── X_train_final.csv     

│    ├── X_test_final.csv 

│    ├── y_train.csv              

│    └── y_test.csv              

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

| Variable | Signification |
| :--- | :--- |
| `Age` | Âge du patient (en années) |
| `Sex` | Sexe (M = Homme, F = Femme) |
| `ChestPainType` | Type de douleur thoracique (TA: Angine typique, ATA: Angine atypique, NAP: Douleur non angineuse, ASY: Asymptomatique) |
| `RestingBP` | Pression artérielle au repos (en mm Hg) |
| `Cholesterol` | Taux de cholestérol sérique (en mg/dl) |
| `FastingBS` | Glycémie à jeun (> 120 mg/dl => 1, sinon 0) |
| `RestingECG` | Résultats électrocardiographiques au repos (Normal, ST, LVH) |
| `MaxHR` | Fréquence cardiaque maximale atteinte |
| `ExerciseAngina` | Angine de poitrine induite par l'effort (Y = Oui, N = Non) |
| `Oldpeak` | Dépression du segment ST par rapport au repos |
| `ST_Slope` | Pente du segment ST lors de l'effort (Up, Flat, Down) |

## 🧠 Modèles testés
- Régression Logistique
- SVM
- Random Forest
- XGBoost

  ## ⚙️ Méthodologie suivie

| Phase | Contenu |
| :--- | :--- |
| **compréhension du problème** | Problème de prédiction des maladies cardiaques, analyse du dataset de 918 patients, objectif de classification binaire (0 = Sain, 1 = Malade). |
| **compréhension de données** | Analyse exploratoire (EDA) des 11 variables cliniques, étude des distributions, corrélations et valeurs manquantes (imputation des valeurs manquantes par algorithme de KNN) |
| **Preparation de données** | division des données en ensembles d'entraînement et de test avec stratification, normalisation des données numériques via `StandardScaler`et Encodage des variables catégorielles (One-Hot Encoding). |
| **entrainement du modèle** | Entraînement et optimisation de 4 modèles : Régression Logistique, SVM, Random Forest et XGBoost. Utilisation de GridSearchCV pour trouver les meilleurs hyperparamètres. |
| **Evaluation** | Évaluation des modèles via la Matrice de Confusion, l'Accuracy et le Recall (Rappel) et Comparaison des performances. |
| **Deployment** | Déploiement de l'application interactive sur Streamlit Cloud, accessible directement en ligne pour tester les prédictions en temps réel. |

### 🏆 Performances (Recall)
| Modèle | Accuracy | Recall |
| :--- | :---: | :---: |
| **Régression Logistique** | 0.87 | 0.87 |
| **SVM** | 0.85 | 0.89 |
| **Random Forest** | 0.85 | 0.85 |
| **XGBoost** | 0.82 | 0.86 |

## 🔗 Tester l'application directement

👉 **[Cliquez ici pour accéder à l'application en ligne](https://heartdiseasepredictionproject.streamlit.app/)**
