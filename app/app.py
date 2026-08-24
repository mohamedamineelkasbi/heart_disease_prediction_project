import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="Prédiction des Maladies Cardiaques",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Prédicteur de Maladie Cardiaque")


@st.cache_resource
def charger_modeles():

    dossier_app = os.path.dirname(os.path.abspath(__file__))
    racine_projet = os.path.dirname(dossier_app)

    scaler_path = os.path.join(racine_projet, "models", "scaler.pkl")
    imputer_path = os.path.join(racine_projet, "models", "imputer.pkl")
    model_svm_path = os.path.join(racine_projet, "models", "best_svm.pkl")
    model_rf_path = os.path.join(racine_projet, "models", "best_rf.pkl")
    model_xgb_path = os.path.join(racine_projet, "models", "best_xgb.pkl")
    model_lr_path = os.path.join(racine_projet, "models", "lr_final.pkl")

    try:
        scaler = joblib.load(scaler_path)
        imputer = joblib.load(imputer_path)
        model_svm = joblib.load(model_svm_path)
        model_rf = joblib.load(model_rf_path)
        model_xgb = joblib.load(model_xgb_path)
        model_lr = joblib.load(model_lr_path)

        modeles = {
            "SVM": model_svm,
            "Random Forest": model_rf,
            "XGBoost": model_xgb,
            "Régression Logistique": model_lr
        }

        return modeles, scaler, imputer

    except FileNotFoundError as e:
        st.error(f"❌ Erreur de chargement des modèles : {e}")
        return None, None, None


modeles, scaler, imputer = charger_modeles()

numerical_cols = [
    'Age',
    'RestingBP',
    'Cholesterol',
    'MaxHR',
    'Oldpeak'
]

tab1, tab2, tab3 = st.tabs([
    "Prédiction",
    "Prédiction en masse",
    "Informations sur les Modèles"
])


with tab1:

    st.sidebar.header("Paramètres du Patient")

    Age = st.sidebar.slider("Âge", 1, 100, 50)
    RestingBP = st.sidebar.slider(
        "Pression artérielle au repos (mm Hg)",
        80, 200, 120
    )
    Cholesterol = st.sidebar.slider(
        "Cholestérol (mg/dl)",
        0, 600, 200
    )
    MaxHR = st.sidebar.slider(
        "Fréquence cardiaque maximale",
        60, 220, 150
    )
    Oldpeak = st.sidebar.slider(
        "Dépression ST",
        -2.6, 7.0, 1.0
    )

    st.sidebar.markdown("---")

    Sex = st.sidebar.selectbox(
        "Sexe",
        ["Homme", "Femme"]
    )

    FastingBS = st.sidebar.selectbox(
        "Glycémie à jeun > 120 mg/dl ?",
        ["Non", "Oui"]
    )

    ExerciseAngina = st.sidebar.selectbox(
        "Angine induite par l'exercice ?",
        ["Non", "Oui"]
    )

    ChestPainType = st.sidebar.selectbox(
        "Type de douleur thoracique",
        [
            "Aucune / ASY",
            "Typique (TA)",
            "Atypique (ATA)",
            "Non angineuse (NAP)"
        ]
    )

    RestingECG = st.sidebar.selectbox(
        "Résultats ECG au repos",
        [
            "Normal",
            "Anomalie ST-T",
            "Hypertrophie ventriculaire (LVH)"
        ]
    )

    ST_Slope = st.sidebar.selectbox(
        "Pente du segment ST à l'effort",
        [
            "Ascendante (Up)",
            "Plate (Flat)",
            "Descendante (Down)"
        ]
    )

    if st.sidebar.button("Lancer la Prédiction"):

        donnees_patient = pd.DataFrame({
            'Age': [Age],
            'RestingBP': [RestingBP],
            'Cholesterol': [Cholesterol],
            'FastingBS': [1 if FastingBS == "Oui" else 0],
            'MaxHR': [MaxHR],
            'Oldpeak': [Oldpeak],
            'Sex': ['M' if Sex == "Homme" else 'F'],
            'ChestPainType': [{
                "Aucune / ASY": "ASY",
                "Typique (TA)": "TA",
                "Atypique (ATA)": "ATA",
                "Non angineuse (NAP)": "NAP"
            }[ChestPainType]],
            'RestingECG': [{
                "Normal": "Normal",
                "Anomalie ST-T": "ST",
                "Hypertrophie ventriculaire (LVH)": "LVH"
            }[RestingECG]],
            'ExerciseAngina': [
                'Y' if ExerciseAngina == "Oui" else 'N'
            ],
            'ST_Slope': [{
                "Ascendante (Up)": "Up",
                "Plate (Flat)": "Flat",
                "Descendante (Down)": "Down"
            }[ST_Slope]]
        })

        donnees_patient['RestingBP'] = (
            donnees_patient['RestingBP'].replace(0, np.nan)
        )

        donnees_patient['Cholesterol'] = (
            donnees_patient['Cholesterol'].replace(0, np.nan)
        )

        donnees_patient[numerical_cols] = scaler.transform(
            donnees_patient[numerical_cols]
        )

        donnees_patient[numerical_cols] = imputer.transform(
            donnees_patient[numerical_cols]
        )

        cols_categorielles = [
            'Sex',
            'ChestPainType',
            'RestingECG',
            'ExerciseAngina',
            'ST_Slope'
        ]

        donnees_patient = pd.get_dummies(
            donnees_patient,
            columns=cols_categorielles,
            drop_first=True,
            dtype=int
        )

        modele_reference = next(iter(modeles.values()))

        donnees_patient = donnees_patient.reindex(
            columns=modele_reference.feature_names_in_,
            fill_value=0
        )

        st.subheader("🔍 Résultats des prédictions")

        for nom_modele, modele in modeles.items():

            prediction = modele.predict(
                donnees_patient
            )[0]

            if prediction == 1:
                st.error(
                    f"🚨 {nom_modele} : Maladie cardiaque détectée"
                )
            else:
                st.success(
                    f"✅ {nom_modele} : Pas de maladie cardiaque"
                )

    else:
        st.info(
            "📌 Remplissez vos informations médicales à gauche "
            "et cliquez sur le bouton."
        )


with tab2:

    st.header("Importer un fichier CSV brut")

    st.write(
        "Veuillez importer votre fichier contenant les données "
        "des patients pour obtenir une prédiction automatique."
    )

    colonnes_requises = [
        'Age',
        'Sex',
        'ChestPainType',
        'RestingBP',
        'Cholesterol',
        'FastingBS',
        'RestingECG',
        'MaxHR',
        'ExerciseAngina',
        'Oldpeak',
        'ST_Slope'
    ]

    fichier_csv = st.file_uploader(
        "Glissez-déposez votre fichier ici",
        type=["csv"]
    )

    st.caption("""
    📝 **Note :** Le fichier doit contenir exactement ces 11 colonnes
    (ordre non imposé) :
    `Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
    RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope`.
    Les valeurs manquantes sont prises en charge automatiquement
    par l'imputation KNN.
    """)

    if fichier_csv is not None:

        df_brut = pd.read_csv(fichier_csv)

        modele_choisi = st.selectbox(
            "Choisir le modèle",
            list(modeles.keys())
        )

        if st.button("🔍 Lancer la prédiction en masse"):

            if set(df_brut.columns) != set(colonnes_requises):
                st.error(
                    "❌ Erreur : les colonnes du fichier "
                    "ne correspondent pas aux attentes."
                )
                st.stop()

            df_brut['RestingBP'] = (
                df_brut['RestingBP'].replace(0, np.nan)
            )

            df_brut['Cholesterol'] = (
                df_brut['Cholesterol'].replace(0, np.nan)
            )

            with st.spinner("Prétraitement des données..."):

                df_encode = df_brut.copy()

                df_encode[numerical_cols] = scaler.transform(
                    df_encode[numerical_cols]
                )

                df_encode[numerical_cols] = imputer.transform(
                    df_encode[numerical_cols]
                )

                cols_categorielles = [
                    'Sex',
                    'ChestPainType',
                    'RestingECG',
                    'ExerciseAngina',
                    'ST_Slope'
                ]

                df_encode = pd.get_dummies(
                    df_encode,
                    columns=cols_categorielles,
                    drop_first=True,
                    dtype=int
                )

            modele = modeles[modele_choisi]

            df_encode = df_encode.reindex(
                columns=modele.feature_names_in_,
                fill_value=0
            )

            predictions = modele.predict(df_encode)

            df_resultats = df_brut.copy()

            df_resultats[ 'Prédiction (0=Sain, 1=Malade)'] = predictions

            st.success(
                "✅ Fichier validé ! Prédiction terminée."
            )

            st.subheader("Résultats :")

            st.dataframe(df_resultats)

            csv = df_resultats.to_csv(
                index=False
            ).encode('utf-8')

            st.download_button(
                label="📥 Télécharger les prédictions (CSV)",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv'
            )


with tab3:

    st.header("Informations sur les Modèles")

    st.subheader("Comparaison des performances (Recall)")

    donnees_graphique = pd.DataFrame({
        'Modèle': [
            'Régression Logistique',
            'Forêt Aléatoire',
            'SVM',
            'XGBoost'
        ],
        'Recall': [
            0.87,
            0.85,
            0.89,
            0.86
        ]
    })

    st.bar_chart(
        donnees_graphique,
        x='Modèle',
        y='Recall',
        height=400
    )
