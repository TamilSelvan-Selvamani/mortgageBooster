{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "6df875c1-5b30-430b-bc75-e585546cd929",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "48d4f0b8-4620-4978-a6b0-3a390d12ea76",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load models and reference dataframe\n",
    "clf = joblib.load(\"C:/Users/Ashok Kumar/risk_classifier.pkl\")\n",
    "regressor = joblib.load(\"C:/Users/Ashok Kumar/loan_regressor.pkl\")\n",
    "reference_df = pd.read_csv(\"C:/Users/Ashok Kumar/reference_data.csv\")  # Preprocessed df used during training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "78feb7cd-fc00-4f59-8d10-532b4f6ac4ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prediction function\n",
    "def predict_borrower(user_input):\n",
    "    input_df = pd.DataFrame([user_input])\n",
    "    full_df = pd.concat([reference_df.drop(['loan_status', 'loan_amnt'], axis=1), input_df], axis=0)\n",
    "    full_encoded = pd.get_dummies(full_df)\n",
    "    full_encoded = full_encoded.reindex(columns=clf.feature_names_in_, fill_value=0)\n",
    "    \n",
    "    risk = clf.predict(full_encoded.tail(1))[0]\n",
    "    loan_amt = regressor.predict(full_encoded.tail(1))[0]\n",
    "\n",
    "    return int(risk), round(loan_amt, 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "0c39ad7b-5c2b-4e59-b07c-c6b11cf5e8ff",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-04-22 21:11:59.365 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\Ashok Kumar\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Streamlit UI\n",
    "st.title(\"Borrower Risk & Loan Amount Predictor\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "e654b06c-aaf4-403c-9594-ee0ee2346847",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-04-22 21:12:36.619 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "# Input form\n",
    "with st.form(\"borrower_form\"):\n",
    "    term = st.selectbox(\"Loan Term\", [' 36 months', ' 60 months'])\n",
    "    grade = st.selectbox(\"Grade\", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])\n",
    "    home_ownership = st.selectbox(\"Home Ownership\", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])\n",
    "    annual_inc = st.number_input(\"Annual Income\", min_value=10000, step=1000)\n",
    "    verification_status = st.selectbox(\"Verification Status\", ['Verified', 'Not Verified', 'Source Verified'])\n",
    "    purpose = st.selectbox(\"Purpose\", ['credit_card', 'debt_consolidation', 'home_improvement', 'major_purchase'])\n",
    "    dti = st.slider(\"Debt-to-Income Ratio (DTI)\", 0.0, 40.0, step=0.1)\n",
    "    open_acc = st.number_input(\"Open Accounts\", min_value=0)\n",
    "    pub_rec = st.number_input(\"Public Records\", min_value=0)\n",
    "    revol_util = st.slider(\"Revolving Utilization (%)\", 0.0, 150.0, step=0.1)\n",
    "    total_acc = st.number_input(\"Total Accounts\", min_value=1)\n",
    "    initial_list_status = st.selectbox(\"Initial List Status\", ['w', 'f'])\n",
    "    application_type = st.selectbox(\"Application Type\", ['Individual', 'Joint App'])\n",
    "\n",
    "    submitted = st.form_submit_button(\"Predict\")\n",
    "\n",
    "if submitted:\n",
    "    user_input = {\n",
    "        'term': term,\n",
    "        'grade': grade,\n",
    "        'home_ownership': home_ownership,\n",
    "        'annual_inc': annual_inc,\n",
    "        'verification_status': verification_status,\n",
    "        'purpose': purpose,\n",
    "        'dti': dti,\n",
    "        'open_acc': open_acc,\n",
    "        'pub_rec': pub_rec,\n",
    "        'revol_util': revol_util,\n",
    "        'total_acc': total_acc,\n",
    "        'initial_list_status': initial_list_status,\n",
    "        'application_type': application_type\n",
    "    }\n",
    "\n",
    "    risk, max_loan = predict_borrower(user_input)\n",
    "    \n",
    "    st.markdown(\"### 🧠 Prediction\")\n",
    "    st.write(\"🔴 **High Risk**\" if risk else \"🟢 **Low Risk**\")\n",
    "    st.write(f\"💰 **Recommended Loan Amount**: ${max_loan}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
