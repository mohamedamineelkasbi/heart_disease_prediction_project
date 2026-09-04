import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Prédiction Maladie Cardiaque", page_icon="❤️", layout="wide")
st.title("❤️ Prédicteur de Maladie Cardiaque")

@st.cache_resource
def charger_modeles():
    dossier_app = os.path.dirname(os.path.abspath(__file__))
    racine = os.path.dirname(dossier_app)
    
    try:
        scaler = joblib.load(os.path.join(racine, "models", "scaler.pkl"))
        imputer = joblib.load(os.path.join(racine, "models", "imputer.pkl"))
        
        modeles = {
            "Régression Logistique": joblib.load(os.path.join(racine, "models", "lr_final.pkl")),
            "Random Forest": joblib.load(os.path.join(racine, "models", "best_rf.pkl")),
            "SVM": joblib.load(os.path.join(racine, "models", "best_svm.pkl")),
            "XGBoost": joblib.load(os.path.join(racine, "models", "best_xgb.pkl"))
        }
        
        X_train_ref = pd.read_csv(os.path.join(racine, "data", "processed", "X_train.csv"))
        reference_columns = X_train_ref.columns.tolist()
        
        return modeles, scaler, imputer, reference_columns
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None, None, None, None

modeles, scaler, imputer, reference_columns = charger_modeles()

if modeles is None:
    st.stop()

raw_num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

def preprocess_infaillible(df_brut):
    df = df_brut.copy()

    df['RestingBP'] = df['RestingBP'].replace(0, np.nan)
    df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)

    df[raw_num_cols] = scaler.transform(df[raw_num_cols])

    df[raw_num_cols] = imputer.transform(df[raw_num_cols])
    
    df_process = pd.DataFrame(0, index=df.index, columns=reference_columns)

    for col in raw_num_cols:
        if col in reference_columns:
            df_process[col] = df[col]
            
    df_process['FastingBS'] = df['FastingBS']

    if 'Sex' in df.columns:
        df_process.loc[df['Sex'] == 'M', 'Sex_M'] = 1
        
    if 'ChestPainType' in df.columns:
        df_process.loc[df['ChestPainType'] == 'ATA', 'ChestPainType_ATA'] = 1
        df_process.loc[df['ChestPainType'] == 'NAP', 'ChestPainType_NAP'] = 1
        df_process.loc[df['ChestPainType'] == 'TA', 'ChestPainType_TA'] = 1
        
    if 'RestingECG' in df.columns:
        df_process.loc[df['RestingECG'] == 'Normal', 'RestingECG_Normal'] = 1
        df_process.loc[df['RestingECG'] == 'ST', 'RestingECG_ST'] = 1
        
    if 'ExerciseAngina' in df.columns:
        df_process.loc[df['ExerciseAngina'] == 'Y', 'ExerciseAngina_Y'] = 1
        
    if 'ST_Slope' in df.columns:
        df_process.loc[df['ST_Slope'] == 'Flat', 'ST_Slope_Flat'] = 1
        df_process.loc[df['ST_Slope'] == 'Up', 'ST_Slope_Up'] = 1

    return df_process[reference_columns]

tab1, tab2, tab3 = st.tabs(["🔍 Prédiction", "📂 Prédiction en masse", "ℹ️ Modèles"])

with tab1:
    st.sidebar.header("Paramètres du Patient")
    
    Age = st.sidebar.slider("Âge", 28, 77, 50)
    RestingBP = st.sidebar.slider("Pression artérielle (mm Hg)", 90, 200, 120)
    Cholesterol = st.sidebar.slider("Cholestérol (mg/dl)", 0, 600, 200)
    FastingBS = st.sidebar.selectbox("Glycémie > 120 ?", [0, 1], format_func=lambda x: "Oui" if x==1 else "Non")
    MaxHR = st.sidebar.slider("Fréquence cardiaque max", 60, 202, 150)
    Oldpeak = st.sidebar.slider("Dépression ST", -2.6, 6.2, 1.0, step=0.1)
    
    st.sidebar.markdown("---")
    Sex = st.sidebar.selectbox("Sexe", ["M", "F"], format_func=lambda x: "Homme" if x=="M" else "Femme")
    ChestPainType = st.sidebar.selectbox(
    "Douleur thoracique",
    ["ASY", "ATA", "NAP", "TA"],
    format_func=lambda x: {
        "ASY": "ASY (Asymptomatique)",
        "ATA": "ATA (Angine atypique)",
        "NAP": "NAP (Douleur non angineuse)",
        "TA": "TA (Angine typique)"
    }[x]
)
    RestingECG = st.sidebar.selectbox(
    "ECG au repos",
    ["Normal", "ST", "LVH"],
    format_func=lambda x: {
        "Normal": "Normal",
        "ST": "ST (Anomalie de l'onde ST-T)",
        "LVH": "LVH (Hypertrophie Ventriculaire Gauche)"
    }[x]
)
    ExerciseAngina = st.sidebar.selectbox("Angine d'effort", ["N", "Y"], format_func=lambda x: "Oui" if x=="Y" else "Non")
    ST_Slope = st.sidebar.selectbox("Pente ST", ["Up", "Flat", "Down"], format_func=lambda x: {"Up":"Ascendante","Flat":"Plate","Down":"Descendante"}[x])

    if st.sidebar.button("🩺 Lancer la Prédiction"):
        df_brut = pd.DataFrame([{
            'Age': Age, 'RestingBP': RestingBP, 'Cholesterol': Cholesterol, 
            'FastingBS': FastingBS, 'MaxHR': MaxHR, 'Oldpeak': Oldpeak,
            'Sex': Sex, 'ChestPainType': ChestPainType, 'RestingECG': RestingECG,
            'ExerciseAngina': ExerciseAngina, 'ST_Slope': ST_Slope
        }])

        df_prep = preprocess_infaillible(df_brut)
        
        st.subheader("🔍 Résultats des prédictions")
        cols = st.columns(4)
        
        for idx, (nom, modele) in enumerate(modeles.items()):
            pred = modele.predict(df_prep)[0]
            with cols[idx]:
                if pred == 1:
                    st.error(f"🚨 **{nom}** : Maladie détectée")
                else:
                    st.success(f"✅ **{nom}** : Pas de maladie")

with tab2:
    st.header("📂 Importer un fichier CSV")
    
    colonnes_requises = ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
                         'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']

    fichier_csv = st.file_uploader("Glissez-déposez votre fichier ici", type=["csv"])
    
    if fichier_csv is not None:
        df_brut_csv = pd.read_csv(fichier_csv)
        modele_choisi = st.selectbox("Choisir le modèle pour la prédiction", list(modeles.keys()))
        
        if st.button("🔍 Lancer la prédiction en masse"):
   
            if set(df_brut_csv.columns) != set(colonnes_requises):
                st.error("❌ Erreur : Les noms de colonnes du fichier ne correspondent pas exactement aux attentes.")
                st.stop()
                
            if df_brut_csv.isnull().values.any():
                st.error("❌ Erreur : Le fichier contient des valeurs vides (NaN). Veuillez nettoyer le fichier.")
                st.stop()
            
            df_prep_csv = preprocess_infaillible(df_brut_csv)
            
            modele = modeles[modele_choisi]
            predictions = modele.predict(df_prep_csv)
            
            df_resultats = df_brut_csv.copy()
            df_resultats['Prédiction (0=Sain, 1=Malade)'] = predictions
            
            st.success("✅ Prédictions terminées avec succès !")
            st.dataframe(df_resultats)
            
            csv_resultat = df_resultats.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les résultats (CSV)",
                data=csv_resultat,
                file_name='predictions_cardiaques.csv',
                mime='text/csv'
            )

    st.markdown("---")
    st.markdown("""
 **📌 NOTE** : Pour garantir des prédictions fiables, le fichier CSV doit contenir **exactement les 11 colonnes suivantes** (ordre non imposé) :
 `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`.
 il ne doit comporter **aucune valeur manquante (NaN)** et doit respecter les formats attendus pour chaque variable, toute valeur aberrante ou illogique peut dégrader les performances du modèle. Le système bloque toute entrée non conforme.
    """)


with tab3:
    st.header("ℹ️ Informations sur les Modèles")
    st.write("Ce tableau récapitule les performances des modèles évalués lors de la phase de test, optimisés spécifiquement sur le **Recall** pour minimiser les faux négatifs en milieu médical.")
    
    donnees_graphique = pd.DataFrame({
        'Modèle': ['Régression Logistique', 'Random Forest', 'SVM', 'XGBoost'],
        'Recall (Sensibilité)': [0.87, 0.85, 0.89, 0.86]
    })
    
    st.bar_chart(donnees_graphique, x='Modèle', y='Recall (Sensibilité)', height=400)
    
    st.markdown("---")
    st.subheader("Pourquoi le Recall ?")
    st.info("""
    En cardiologie, un **Faux Négatif** (dire à un patient malade qu'il est sain) est beaucoup plus dangereux 
    qu'un **Faux Positif** (dire à un patient sain qu'il est malade, ce qui nécessite juste un examen complémentaire). 
    Nous avons donc configuré les algorithmes pour maximiser leur capacité à détecter les vrais malades.
    """)
