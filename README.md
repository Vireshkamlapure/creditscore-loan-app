# 🚀 CrediScore: Real-Time Loan Approval Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://creditscore-loan-app-wasjdrjvzptdgjjghc8ai2.streamlit.app/)

This project is a complete, end-to-end Machine Learning web application deployed on Streamlit Community Cloud. It uses a `scikit-learn` model to predict whether a loan application will be approved or rejected based on the applicant's financial and personal details.

**Visit the live application:** [https://creditscore-loan-app.streamlit.app/](https://creditscore-loan-app-wasjdrjvzptdgjjghc8ai2.streamlit.app/)

---

## 📸 Application Screenshot

Here is a preview of the application's user interface. The user enters the applicant's details in the form, and the model provides an instant prediction.

### Prediction: Approved
A demonstration of a successful loan application prediction.

![Approved Case](./approved_case.png)

### Prediction: Rejected
A demonstration of an unsuccessful loan application prediction.

![Rejected Case](./rejected_case.png)

---

## 💡 About the Project

This app was built as an end-to-end project to demonstrate the entire data science lifecycle: from data exploration and model training to building a fully functional, interactive web application.

The app, "CrediScore," acts as a tool for a finance department. It takes a loan applicant's details as input and uses a pre-trained machine learning model to provide an instant prediction on whether the loan should be "Approved" or "Rejected," along with a confidence score.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Data Analysis:** Pandas, Jupyter Notebook
* **Machine Learning:** Scikit-learn
* **Web Framework:** Streamlit
* **Deployment:** Streamlit Community Cloud & GitHub

---

## 🧠 Project Methodology

The core of this application is the `loan_model.joblib` file, which is a saved `scikit-learn` pipeline.

1.  **Data Exploration:** The [Kaggle Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) was used.
2.  **Preprocessing:** A `ColumnTransformer` pipeline was built to handle:
    * **Numerical Features:** Missing values were filled with the **median**, and data was scaled using `StandardScaler`.
    * **Categorical Features:** Missing values were filled with the **most frequent** value, and features were one-hot encoded using `OneHotEncoder`.
3.  **Model Training:** A `LogisticRegression` model was trained on the preprocessed data. This model was chosen for its interpretability and solid baseline performance (~79% accuracy).
4.  **Deployment:** The final, trained pipeline (which includes both the preprocessor and the model) was saved as a single `joblib` file. The Streamlit app loads this file to make live predictions on new, unseen data.

---

## ⚙️ How to Run Locally

Want to run this project on your own machine?

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/creditscore-loan-app.git](https://github.com/YOUR_USERNAME/creditscore-loan-app.git)
    cd creditscore-loan-app
    ```
    *(Replace `YOUR_USERNAME` with your GitHub username)*

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

---

## 🧑‍💻 Author

* **Viresh Kamlapure**