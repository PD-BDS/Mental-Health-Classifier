# 🧠 Mental Health Status Classifier

This project is an NLP-based web application that classifies textual inputs into seven distinct mental health categories using a fine-tuned BERT model. It helps raise awareness and support early detection of mental health signals in user-generated texts.

🚀 **Live Demo on Hugging Face Spaces**  
👉 [piyaldey/Mental_Health_Status_Classifier](https://huggingface.co/spaces/piyaldey/Mental_Health_Status_Classifier)

---

## 🧩 Features

- 🔍 Classifies text into one of the following:
  - Normal
  - Depression
  - Suicidal
  - Anxiety
  - Bipolar
  - Stress
  - Personality Disorder
- 🧠 Powered by `bert-base-uncased` from Hugging Face Transformers
- 💬 Clean, minimal UI built with Streamlit
- 🖥️ Deployable on Hugging Face Spaces, Streamlit Cloud, or locally

---

## 📊 Dataset

The model was trained on a rich dataset combining multiple Kaggle mental health-related sources:

- Depression Reddit Cleaned
- Mental Health Dataset Bipolar
- Suicidal Tweet Detection Dataset
- Students Anxiety and Depression Dataset
- and others...

All merged and preprocessed into a clean, labeled format with 7 mental health categories.

📂 Dataset Link: [Kaggle Mental Health Dataset](https://www.kaggle.com/code/swarnabh31/nlp-bert-sentimentanalysis-mentalhealth)

---

## 🧠 Model Details

- Model: `bert-base-uncased` fine-tuned with a classification head
- Loss Function: CrossEntropyLoss
- Optimizer: AdamW
- Epochs: 4
- Evaluation Metrics: Accuracy, Precision, Recall, F1-Score

The model was trained on a balanced dataset using PyTorch and Hugging Face `transformers` library.

---

## 🖥️ App Preview

![App Preview](https://huggingface.co/spaces/piyaldey/Mental_Health_Status_Classifier/resolve/main/preview.png)

---

## ⚙️ Installation & Usage

### 🔧 Requirements

```bash
pip install streamlit torch transformers
```

### ▶️ Run Locally

1. Clone the repo:
    ```bash
    git clone https://github.com/PD-BDS/Mental-Health-Classifier.git
    cd Mental-Health-Classifier
    ```

2. Place the trained model (`best_model.pth`) in the root directory.

3. Launch the app:
    ```bash
    streamlit run app.py
    ```

---

## 📁 Project Structure

```
Mental_Health_Status_Classifier/
│
├── app.py                            # Streamlit frontend
├── best_model.pth                    # Trained model (not included in repo, only available in huggingface)
├── mental_health_classifier.ipynb     # Notebook for preprocessing and model training
└── README.md               
```

---

## 📦 Deployment

This app is deployed on Hugging Face Spaces using their `streamlit` SDK. You can deploy it on:

- Hugging Face Spaces (recommended)
- Streamlit Cloud
- Docker (optional)

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Kaggle Mental Health Datasets](https://www.kaggle.com)
- [Streamlit](https://streamlit.io/)
