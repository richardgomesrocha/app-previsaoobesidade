import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(page_title="Análise de Obesidade", layout="wide")
st.title("⚖️ Previsão de Obesidade - Mini ML App")
st.write("Envie um CSV com dados de saúde ou use o dataset de exemplo.")

# =========================
# UPLOAD DO CSV
# =========================
uploaded_file = st.file_uploader("📂 Envie seu CSV", type=["csv"])
# Dataset padrão se nenhum CSV enviado
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Usando dataset de exemplo.")
    dados = {
        'Age': [25, 30, 22, 40, 35, 28, 50, 45],
        'Height': [170, 165, 180, 160, 175, 168, 172, 169],
        'Weight': [68, 85, 95, 70, 110, 55, 120, 80],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'Obesity': [
            'Normal_Weight',
            'Overweight_Level_I',
            'Obesity_Type_I',
            'Normal_Weight',
            'Obesity_Type_II',
            'Underweight',
            'Obesity_Type_III',
            'Overweight_Level_II'
        ]
    }
    df = pd.DataFrame(dados)

st.subheader("📊 Visualização do dataset")
st.dataframe(df.head())

# =========================
# PREPARAÇÃO DOS DADOS
# =========================
colunas = ['Age', 'Height', 'Weight', 'Gender', 'Obesity']
df = df[colunas].dropna()

# Conversão de variáveis numéricas
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Height'] = pd.to_numeric(df['Height'], errors='coerce')
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
df = df.dropna()

# Codificação de categóricas
encoders = {}
for col in ['Gender', 'Obesity']:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

X = df[['Age', 'Height', 'Weight', 'Gender']]
y = df['Obesity']

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# TREINAMENTO DOS MODELOS
# =========================
modelos = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'LR': LogisticRegression(max_iter=1000, random_state=42)
}

acuracias = {}
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    preds = modelo.predict(X_test)
    acuracias[nome] = accuracy_score(y_test, preds)

# Seleção do melhor modelo
melhor_modelo_nome = max(acuracias, key=acuracias.get)
melhor_modelo = modelos[melhor_modelo_nome]

st.subheader("🎯 Acurácia dos modelos")
st.write({k: f"{v:.2%}" for k, v in acuracias.items()})
st.success(f"🏆 Melhor modelo: {melhor_modelo_nome}")

# =========================
# ENTRADA DO USUÁRIO
# =========================
st.subheader("🧠 Previsão de Obesidade - Insira seus dados")

idade = st.slider("Idade", 0, 100, 25)
altura = st.slider("Altura (cm)", 100, 220, 170)
peso = st.slider("Peso (kg)", 30, 200, 70)
genero = st.selectbox("Gênero", list(encoders['Gender'].classes_))
genero_idx = encoders['Gender'].transform([genero])[0]

entrada = np.array([[idade, altura, peso, genero_idx]])

if st.button("🔍 Prever Obesidade"):
    pred = melhor_modelo.predict(entrada)[0]
    classe = encoders['Obesity'].inverse_transform([pred])[0]
    st.write(f"⚖️ **Resultado da análise:** {classe}")
    st.write(f"📈 **Modelo utilizado:** {melhor_modelo_nome}")
