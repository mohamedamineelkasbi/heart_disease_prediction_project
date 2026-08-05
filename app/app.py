import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="Prédiction Maladie Cardiaque", page_icon="❤️", layout="wide")
st.title("❤️ Prédicteur de Maladie Cardiaque")


@st.cache_resource
def charger_modeles():

    dossier_app = os.path.dirname(os.path.abspath(__file__))
    
    racine_projet = os.path.dirname(dossier_app)

    scaler_path = os.path.join(racine_projet, "models", "scaler.pkl")
    model_dt_path = os.path.join(racine_projet, "models", "best_dt.pkl")
    model_rf_path = os.path.join(racine_projet, "models", "best_rf.pkl")
    model_xgb_path = os.path.join(racine_projet, "models", "best_xgb.pkl")
    model_lr_path = os.path.join(racine_projet, "models", "lr_final.pkl")

    try:
        
        scaler = joblib.load(scaler_path)
        model_dt = joblib.load(model_dt_path)
        model_rf = joblib.load(model_rf_path)
        model_xgb = joblib.load(model_xgb_path)
        model_lr = joblib.load(model_lr_path)

        modeles = {
            "Decision Tree": model_dt,  
            " Random Forest": model_rf,
            "XGBoost": model_xgb, 
            " Régression Logistique": model_lr   
        }

        return modeles, scaler

    except FileNotFoundError as e:
        st.error(f"❌ Erreur de chargement des modèles : {e}. Vérifiez que les fichiers .pkl sont bien présents dans le dossier 'models' à la racine.")
        return None, None

modeles, scaler = charger_modeles()

numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

tab1, tab2, tab3 = st.tabs(["Prédiction", "Prédiction en masse", "Informations Modèles"])


with tab1:
    st.sidebar.header("Paramètres du Patient")

    Age = st.sidebar.slider("Âge", 1, 100, 50)
    RestingBP = st.sidebar.slider("Pression artérielle au repos (mm Hg)", 80, 200, 120)
    Cholesterol = st.sidebar.slider("Cholestérol (mg/dl)", 0, 600, 200)
    MaxHR = st.sidebar.slider("Fréquence cardiaque maximale", 60, 220, 150)
    Oldpeak = st.sidebar.slider("Dépression ST", -2.6, 7.0, 1.0)

    st.sidebar.markdown("---")

    Sex = st.sidebar.selectbox("Sexe", ["Homme", "Femme"])
    FastingBS = st.sidebar.selectbox("Glycémie à jeun > 120 mg/dl ?", ["Non", "Oui"])
    ExerciseAngina = st.sidebar.selectbox("Angine induite par l'exercice ?", ["Non", "Oui"])
    
    ChestPainType = st.sidebar.selectbox("Type de douleur thoracique", 
                                         ["Aucune / ASY", "Typique (TA)", "Atypique (ATA)", "Non angineuse (NAP)"])
    RestingECG = st.sidebar.selectbox("Résultats ECG au repos", 
                                      ["Normal", "Anomalie ST-T", "Hypertrophie ventriculaire (LVH)"])
    ST_Slope = st.sidebar.selectbox("Pente du segment ST à l'effort", 
                                    ["Ascendante (Up)", "Plate (Flat)", "Descendante (Down)"])

    if st.sidebar.button("Lancer la Prédiction"):
        
        donnees_patient = pd.DataFrame({
            'Age': [Age],
            'RestingBP': [RestingBP],
            'Cholesterol': [Cholesterol],
            'FastingBS': [1 if FastingBS == "Oui" else 0],
            'MaxHR': [MaxHR],
            'Oldpeak': [Oldpeak],
            
            'Sex_M': [1 if Sex == "Homme" else 0],
            
            'ChestPainType_ATA': [1 if ChestPainType == "Atypique (ATA)" else 0],
            'ChestPainType_NAP': [1 if ChestPainType == "Non angineuse (NAP)" else 0],
            'ChestPainType_TA': [1 if ChestPainType == "Typique (TA)" else 0],
            
            'RestingECG_Normal': [1 if RestingECG == "Normal" else 0],
            'RestingECG_ST': [1 if RestingECG == "Anomalie ST-T" else 0],
            
            'ExerciseAngina_Y': [1 if ExerciseAngina == "Oui" else 0],
            
            'ST_Slope_Flat': [1 if ST_Slope == "Plate (Flat)" else 0],
            'ST_Slope_Up': [1 if ST_Slope == "Ascendante (Up)" else 0]
        })

        donnees_patient[numerical_cols] = scaler.transform(donnees_patient[numerical_cols])

        st.subheader("🔍 Résultats des prédictions")

        for nom_modele, modele in modeles.items():
            prediction = modele.predict(donnees_patient)[0]
            
            if prediction == 1:
                st.error(f"🚨 {nom_modele} : Maladie cardiaque détectée")
            else:
                st.success(f"✅ {nom_modele} : Pas de maladie cardiaque")
    else:
        st.info("📌 Remplissez vos informations médicales à gauche et cliquez sur le bouton.")

with tab2:
    st.header("Importer un fichier CSV brut")
    st.write("Veuillez importer votre fichier contenant les données des patients pour obtenir une prédiction automatique.")
    
    colonnes_requises = [
        'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 
        'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
    ]
    
    fichier_csv = st.file_uploader("Glissez-déposez votre fichier ici", type=["csv"])
    
    st.caption("""
    📝 **Note :** Le fichier doit contenir exactement ces 11 colonnes (ordre non imposé) : 
    `Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope`. 
    Vérifier que les données soient sans valeurs manquantes NaN ni valeurs illogiques pour garantir la fiabilité des prédictions.
    """)

    if fichier_csv is not None:
        df_brut = pd.read_csv(fichier_csv)
        modele_choisi = st.selectbox("Choisir le modèle", list(modeles.keys()))
        
        if st.button("🔍 Lancer la prédiction en masse"):
            
            if set(df_brut.columns) != set(colonnes_requises):
                st.error("❌ Erreur : Les colonnes du fichier ne correspondent pas aux attentes. Veuillez vérifier la note ci-dessus.")
                st.stop()
            
            if df_brut.isnull().values.any():
                st.error("❌ Erreur : Le fichier contient des valeurs manquantes (NaN). Veuillez nettoyer votre fichier.")
                st.stop()
            
            with st.spinner("Encodage des variables catégorielles..."):
                df_encode = pd.DataFrame()
                df_encode['Age'] = df_brut['Age']
                df_encode['RestingBP'] = df_brut['RestingBP']
                df_encode['Cholesterol'] = df_brut['Cholesterol']
                df_encode['FastingBS'] = df_brut['FastingBS']
                df_encode['MaxHR'] = df_brut['MaxHR']
                df_encode['Oldpeak'] = df_brut['Oldpeak']
                
                df_encode['Sex_M'] = (df_brut['Sex'] == 'M').astype(int)
                df_encode['ChestPainType_ATA'] = (df_brut['ChestPainType'] == 'ATA').astype(int)
                df_encode['ChestPainType_NAP'] = (df_brut['ChestPainType'] == 'NAP').astype(int)
                df_encode['ChestPainType_TA'] = (df_brut['ChestPainType'] == 'TA').astype(int)
                df_encode['RestingECG_Normal'] = (df_brut['RestingECG'] == 'Normal').astype(int)
                df_encode['RestingECG_ST'] = (df_brut['RestingECG'] == 'ST').astype(int)
                df_encode['ExerciseAngina_Y'] = (df_brut['ExerciseAngina'] == 'Y').astype(int)
                df_encode['ST_Slope_Flat'] = (df_brut['ST_Slope'] == 'Flat').astype(int)
                df_encode['ST_Slope_Up'] = (df_brut['ST_Slope'] == 'Up').astype(int)

            df_encode[numerical_cols] = scaler.transform(df_encode[numerical_cols])

            modele = modeles[modele_choisi]
            predictions = modele.predict(df_encode)
            
            df_brut['Prédiction (0=Sain, 1=Malade)'] = predictions
            
            st.success("✅ Fichier validé ! Prédiction terminée.")
            st.subheader("Résultats :")
            st.dataframe(df_brut)
            
            csv = df_brut.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les prédictions (CSV)",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv'
            )
            
with tab3:
    st.header("Informations sur les Modèles")
    st.subheader("Comparaison des performances (recall)")

    donnees_graphique = pd.DataFrame({
        'Modèle': ['Régression Logistique', 'Forêt Aléatoire', 'Arbre de décision', 'XGBoost'],
        'Accuracy': [0.83, 0.83, 0.85, 0.86]
    })

    st.bar_chart(donnees_graphique, x='Modèle', y='recall', height=400)
