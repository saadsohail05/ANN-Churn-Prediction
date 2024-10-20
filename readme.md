
# 🏦 Bank Customer Churn Prediction


## 🚀 Introduction

The **Bank Customer Churn Prediction** project is a powerful tool designed to help banks predict customer churn using **Artificial Neural Networks (ANN)**. By analyzing customer data such as credit score, age, and balance, this model provides insights into which customers are at risk of leaving, allowing banks to take proactive measures.

This project is integrated into a sleek, interactive web app built with **Streamlit**, giving users an easy way to input customer information and receive real-time churn predictions.

## 📦 Installation

To set up this project locally, follow the steps below:

### Prerequisites
Ensure the following are installed:
- Python 3.x
- Pip
- (Optional) Virtualenv

### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/saadsohail05/bank-churn-prediction.git
   cd bank-churn-prediction
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the following files are present:**
   - `model.h5` (Pre-trained ANN model)
   - `labelencoder.pkl` (Label Encoder for Gender)
   - `onehotencoder.pkl` (One-Hot Encoder for Geography)
   - `scaler.pkl` (Scaler for data normalization)

## 💻 Usage

To launch the application and start predicting customer churn:

```bash
streamlit run app.py
```

You’ll be directed to a local web page where you can input customer data and receive predictions.

### Input Fields:
- **Credit Score** (Range: 300 - 850)
- **Geography** (Options: France, Germany, Spain)
- **Gender** (Options: Male, Female)
- **Age** (Range: 18 - 100)
- **Tenure** (Years with the bank)
- **Balance** (Customer account balance)
- **Number of Products** (Products the customer uses)
- **Has Credit Card** (True/False)
- **Is Active Member** (True/False)
- **Estimated Salary**

Once you provide the data, the model predicts the probability of customer churn with helpful insights.

## ✨ Features

- **🔮 Churn Prediction**: Predicts the likelihood of a customer leaving the bank based on their profile.
- **🖥 Interactive Interface**: Built with Streamlit, offering a user-friendly experience with real-time predictions.
- **📊 Visual Insights**: Includes a probability gauge and feature importance charts to help you interpret the results.
- **📝 Actionable Recommendations**: Based on prediction outcomes, the app suggests strategies for retention.
- **📉 Feature Importance**: Visualizes the contribution of different features in the prediction.

## 📂 Data

The project uses the **Churn_Modelling.csv** dataset, which contains information about 10,000 bank customers. Key features include:

- **CreditScore**: Customer's credit score.
- **Geography**: Customer's location (France, Germany, Spain).
- **Gender**: Male or Female.
- **Age**: Customer's age.
- **Tenure**: Number of years the customer has been with the bank.
- **Balance**: Customer’s account balance.
- **Number of Products**: Number of products the customer is using.
- **HasCrCard**: Does the customer have a credit card?
- **IsActiveMember**: Is the customer an active bank member?
- **Estimated Salary**: Customer's estimated annual salary.
- **Exited**: Did the customer churn?

### Preprocessing Steps:
- **Label Encoding**: Converts categorical variables like Gender to numeric.
- **One-Hot Encoding**: Transforms the Geography column into multiple binary columns.
- **Standardization**: Scales numeric features using \`StandardScaler\`.

## 🧠 Methodology

The model is built using an **Artificial Neural Network (ANN)** with the following architecture:

- **Input Layer**: Takes 11 input features.
- **Hidden Layers**: Two hidden layers with 64 and 32 neurons, respectively, using **ReLU** activation.
- **Output Layer**: A single neuron with **sigmoid** activation for binary classification (churn or no churn).

The model is trained using:
- **Optimizer**: Adam with a learning rate of 0.01.
- **Loss Function**: Binary Cross-Entropy.
- **Early Stopping**: Prevents overfitting by stopping training when validation loss plateaus.

## 📊 Results

- **Accuracy**: The model achieved ~85% accuracy on the test data.
- **Churn Prediction Gauge**: Displays the predicted probability of churn, offering a clear interpretation of risk levels.
- **Feature Importance**: Visualized using bar charts to show which factors have the most influence on churn prediction.

The app offers detailed predictions and recommendations, empowering banks to retain at-risk customers more effectively.

## 🏁 Conclusion

The **Bank Customer Churn Prediction** project is a practical solution for banks looking to understand customer behavior and reduce churn. By leveraging machine learning, the model provides clear, actionable insights that can help improve customer retention.

## 🔮 Future Work

Some areas for future development include:
- **Model Improvement**: Tuning the model further or testing advanced algorithms (e.g., XGBoost, Random Forest).
- **Explainability**: Adding interpretability techniques like **SHAP** to explain model decisions.
- **Real-Time Predictions**: Integrating the app with live bank data for real-time predictions.
- **Additional Features**: Incorporating more customer interaction data (e.g., complaints, transaction history).


## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
