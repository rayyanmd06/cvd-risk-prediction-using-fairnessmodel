💓 Cardiovascular Disease Risk Prediction Dashboard

A full-fledged machine learning pipeline and interactive dashboard to predict the risk of cardiovascular disease using clinical features, with support for fairness-aware modeling.


🚀 Features

* ✅ **Preprocessed medical datasets** (Cleveland & Framingham)
* ✅ **Multiple ML models** (Logistic Regression, Random Forest, XGBoost, MLP)
* ✅ **Model selection & ensemble voting**
* ✅ **Fairness mitigation** using Demographic Parity (Fairlearn)
* ✅ **Interactive dashboard** via Streamlit
* ✅ **Dockerized for deployment**


🗂️ Project Structure


📁 Final Project/
│
├── data/                   # Contains heart_cleveland.csv and heart_framingham.csv
├── results/                # Stores models, scalers, SHAP plots, fairness metrics, etc.
├── src/                    # All Python source scripts (training, metrics, preprocessing)
├── assets/                 # Optional logos or images used in the dashboard
├── streamlit_app.py        # Streamlit dashboard file
├── Dockerfile              # For containerizing the app
└── README.md               # Project documentation


📊 Dataset

* **Cleveland Heart Disease Dataset**
* **Framingham Heart Study Dataset**
* Combined and harmonized for richer features.


🧠 Models Trained

| Model               | Accuracy | ROC-AUC | Notes                          |
| ------------------- | -------- | ------- | ------------------------------ |
| Logistic Regression | \~71%    | \~76%   | Baseline                       |
| Random Forest       | \~76%    | \~74%   | Good recall                    |
| XGBoost             | \~81%    | \~72%   | Best performing standard model |
| MLP Classifier      | \~78%    | \~71%   | Neural approach                |
| Voting Ensemble     | \~79%    | \~75%   | Combines top 3 models          |
| **Fair Model**      | \~85%    | \~58%   | With Demographic Parity (DP)   |


🎯 Fairness Evaluation

* 📉 **Demographic Parity Difference (DPD)** reduced to `0.014`
* 📉 **Equalized Odds Difference (EOD)** reduced to `0.071`
* 📌 Fair model trained with `ExponentiatedGradient` on sex attribute as sensitive feature



🖥️ Streamlit Dashboard

* 📌 Model selection: Choose standard or fair model
* 📌 User-friendly input form for all features
* 📌 Confidence scores for standard model
* 📌 Risk gauge visualization
* 🎉 Random animations on prediction (balloons, snow, etc.)
* 🌒 Dark theme styling with custom CSS




🔒 Privacy & Licensing

This project is **private** and intended for academic purposes. Unauthorized publication or duplication is strictly discouraged.



✍️ Author

M Rayyan
B.Tech – Final Year Student
Email: rayyanmd.professional@gmail.com



