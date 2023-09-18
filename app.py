import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import time
import matplotlib.pyplot as plt


# Definindo opções para os campos do formulário
GENDER_OPTIONS = ['Female', 'Male']
SENIOR_CITIZEN_OPTIONS = ['No', 'Yes']
CUSTOMER_PARTNER_OPTIONS = ['No', 'Yes']
CUSTOMER_DEPENDENTS_OPTIONS = ['No', 'Yes']
PHONE_SERVICE_OPTIONS = ['No', 'Yes']
MULTIPLE_LINES_OPTIONS = ['No', 'Yes', 'No phone service']
INTERNET_SERVICE_OPTIONS = ['DSL', 'Fiber optic', 'No']
ONLINE_SECURITY_OPTIONS = ['No', 'Yes', 'No internet service']
ONLINE_BACKUP_OPTIONS = ['No', 'Yes', 'No internet service']
DEVICE_PROTECTION_OPTIONS = ['No', 'Yes', 'No internet service']
TECH_SUPPORT_OPTIONS = ['No', 'Yes', 'No internet service']
STREAMING_TV_OPTIONS = ['No', 'Yes', 'No internet service']
STREAMING_MOVIES_OPTIONS = ['No', 'Yes', 'No internet service']
CONTRACT_OPTIONS = ['One year', 'Month-to-month', 'Two year']
PAPERLESS_BILLING_OPTIONS = ['Yes', 'No']
PAYMENT_METHOD_OPTIONS = ['Mailed check', 'Electronic check', 'Credit card (automatic)', 'Bank transfer (automatic)']

# Caminho do modelo
MODEL_PATH = './modelos/modelo_producao.pkl'
    
def collect_user_input():
    """Coleta os dados de entrada do usuário e retorna um dicionário."""
    
    col1, col2, col3 = st.columns(3)
    
    data = {}
    
    with col1:
        data['customer_gender'] = st.selectbox('Gender:', GENDER_OPTIONS)
        data['customer_senior_citizen'] = st.selectbox('Senior Citizen:', SENIOR_CITIZEN_OPTIONS)
        data['customer_partner'] = st.selectbox('Customer Partner:', CUSTOMER_PARTNER_OPTIONS)
        data['customer_dependents'] = st.selectbox('Customer Dependents:', CUSTOMER_DEPENDENTS_OPTIONS)
        data['customer_tenure'] = st.number_input('Customer Tenure:', min_value=0, max_value=100, value=18)
        data['phone_service'] = st.selectbox('Phone Service:', PHONE_SERVICE_OPTIONS)
        
    with col2:
        data['multiple_lines'] = st.selectbox('Multiple Lines:', MULTIPLE_LINES_OPTIONS)
        data['internet_service'] = st.selectbox('Internet Service:', INTERNET_SERVICE_OPTIONS)
        data['online_security'] = st.selectbox('Online Security:', ONLINE_SECURITY_OPTIONS)
        data['online_backup'] = st.selectbox('Online Backup:', ONLINE_BACKUP_OPTIONS)
        data['device_protection'] = st.selectbox('Device Protection:', DEVICE_PROTECTION_OPTIONS)
        data['tech_support'] = st.selectbox('Tech Support:', TECH_SUPPORT_OPTIONS)

    with col3:
        data['streaming_tv'] = st.selectbox('Streaming TV:', STREAMING_TV_OPTIONS)
        data['streaming_movies'] = st.selectbox('Streaming Movies:', STREAMING_MOVIES_OPTIONS)
        data['contract'] = st.selectbox('Contract:', CONTRACT_OPTIONS)
        data['paperless_billing'] = st.selectbox('Paperless Billing:', PAPERLESS_BILLING_OPTIONS)
        data['payment_method'] = st.selectbox('Payment Method:', PAYMENT_METHOD_OPTIONS)
        data['monthly_charges'] = st.number_input('Monthly Charges:', min_value=0, max_value=100, value=18)
    
    return data


def main():
    
    # Configurando a página para deixar a logo no centro
    
    coluna1, coluna2, coluna3 = st.columns([1, 10, 1])
    image = Image.open('./identidade_visual/Logo (4).png') 
    with coluna2:
        st.image(image, use_column_width=True, width=900)
      
    st.markdown("<h3 style='text-align: center;'>Previsão de Churn para os clientes da Novexus</h3>", unsafe_allow_html=True)

    input_data = collect_user_input()

    df = pd.DataFrame([input_data.values()], columns=input_data.keys())
    
    rf = pickle.load(open(MODEL_PATH, 'rb'))
    
    dicionario = {'No internet service':0,
                  'No phone service': 0,
                  'No': 0,
                  'Yes': 1,
                  'Male': 0,
                  'Female': 1}

    colunas = ['customer_gender', 'customer_senior_citizen',
       'customer_partner', 'customer_dependents', 'customer_tenure',
       'phone_service', 'multiple_lines',
       'online_security', 'online_backup', 'device_protection', 'tech_support',
       'streaming_tv', 'streaming_movies', 'paperless_billing', 'monthly_charges']

    df[colunas] = df[colunas].replace(dicionario)

    prediction = rf.predict(df)
    
    # extra: probabilidade de churn
    probability = rf.predict_proba(df)
    
    if st.button('Fazer previsão', help='Clique neste botão para realizar a previsão de churn do cliente selecionado.'):
        
        with st.spinner('Carregando...'):
            time.sleep(3)
        
        st.markdown("<h3 style='text-align: center;'>Resultado da previsão</h3>", unsafe_allow_html=True)  
        
        if prediction[0] == 0:
            st.error('Churn: Não', icon="❌")
        else:
            st.success('Churn: Sim', icon="✅")
                
        st.progress(probability[0][1], text=f':black[Probabilidade de churn: {100*probability[0][1]:.2f}%]')
                
if __name__ == "__main__":
    main()
