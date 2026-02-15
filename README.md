# Fake_Email_Detector
A Machine Learning based Fake/Spam Email Detector built using Python, Scikit-learn, and Streamlit.
This project analyzes multiple email-related features and predicts whether an email is Spam (Fake) or Real.

🚀 Features

✅ Detects Spam vs Real Emails using ML
📊 Analytics Dashboard with interactive charts
🎮 Interactive Demo Mode (live feature sliders)
📥 Download Prediction Report + History as CSV
🧠 Random Forest Model with StandardScaler

#🧠 Machine Learning Model

Algorithm Used: Random Forest Classifier

Scaling: StandardScaler

Dataset: Generated synthetic dataset (spam_email.csv)

Target Column: is_spam

📌 Input Features Used
Feature	Description
has_link	Email contains a link
has_money_words	Contains money-related words
has_urgent_words	Contains urgent words
has_caps	Contains capital letters
num_exclamation	Count of !
email_length	Total email length
num_digits	Count of digits
num_special_chars	Count of special characters
📂 Project Structure
fake_email_detection/
│
├── app.py                     # Streamlit web app
├── model.py                   # Model training script
├── email.py                   # Dataset generation script
│
├── data/
│   └── spam_email.csv         # Dataset
│
├── models/
│   ├── fake_email_model.pkl   # Trained model
│   ├── scaler.pkl             # StandardScaler
│   └── feature_columns.pkl    # Feature list
│
└── requirements.txt

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/fake_email_detection.git
cd fake_email_detection

2️⃣ Install Dependencies
pip install -r requirements.txt

🏋️‍♂️ Train the Model

Run this command to generate dataset + train model:

python model.py


This will create:

data/spam_email.csv

models/fake_email_model.pkl

models/scaler.pkl

models/feature_columns.pkl

▶️ Run the Streamlit App
streamlit run app.py

📊 App Pages Included

🏠 Home
🔎 Detect Emails
📊 Analytics Dashboard
🎮 Interactive Demo
ℹ️ About

📥 Outputs

✔ Prediction result (Spam / Real)
✔ Confidence score
✔ Probability distribution
✔ Downloadable report CSV
✔ Prediction history download

🛠️ Technologies Used

Python

Streamlit

Pandas, NumPy

Scikit-learn

Plotly

Joblib
