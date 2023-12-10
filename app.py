import streamlit as st
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset from Google Drive
file_url = 'https://drive.google.com/uc?id=1-0kzTU65PMKfaMFuF0IXbg5KrFAyhE1a'
output_file = 'pre-processing.csv'

# Load the dataset
cirrhosis = pd.read_csv(output_file)

# Memisahkan dataset cirrhosis menjadi fitur (X) dan target (y), dan mendefinisikan nama fitur.
X = cirrhosis.drop(['Status'], axis=1)
y = cirrhosis['Status']
feature_names = X.columns

# Melakukan oversampling pada data menggunakan SMOTE untuk menangani ketidakseimbangan kelas.
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model Random Forest
rf_model = RandomForestClassifier(random_state=42)

# Melatih model menggunakan data latih
rf_model.fit(X_train, y_train)

# Streamlit app
st.title('Cirrhosis Patient Survival Prediction')
st.subheader('Keterangan Input Fitur :')
st.write('Drug')
st.write('0 = Placebo, 1 = D-penicillamine')
st.write('Sex')
st.write('0 = Female, 1 = Male')
st.write('Ascites, Hepatomegaly, Spiders, Edema')
st.write('0 = No, 1 = Yes')
st.write('Stage')
st.write('Tahap Histologis Penyakit')

# User input for feature prediction
st.sidebar.header('Input Fitur')

input_features = {}
for feature in feature_names:
    if feature in ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage']:
        if feature == 'Stage':
            input_features[feature] = st.sidebar.selectbox(f'{feature}', [1, 2, 3, 4])
        else:
            input_features[feature] = st.sidebar.selectbox(f'{feature}', [0, 1])
    else:
        input_features[feature] = st.sidebar.number_input(f'{feature}', float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))

input_df = pd.DataFrame([input_features])

# Predict the status for the user input
prediction = rf_model.predict(input_df)

# Display the prediction
st.subheader('Prediksi')
st.write(f'Status : {prediction[0]}')
st.write('D (meninggal), C (hidup), CL (hidup karena transplantasi hati)')

# Evaluation on the test set
st.subheader('Evaluasi Model Pada Test Set')
# Memprediksi label pada data uji
y_pred = rf_model.predict(X_test)

# Mengukur performa model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Menampilkan hasil evaluasi
st.write(f'Akurasi: {accuracy:.2f}')
st.write(f'Confusion Matrix:\n{conf_matrix}')
st.write(f'Klasifikasi Report:\n{class_report}')
