# 🧠 Modular Fraud Detection Pipeline

A **production-grade, modular fraud detection system** built for **reproducibility, transparency, and scalability**.  
Designed with **MLOps best practices** using **Python, MLflow, Prefect, LightGBM, and Docker**.

---

## 📁 Project Structure

```
├── src/
│   ├── data/
│   │   ├── data_loader.py        # Load raw datasets
│   │   ├── data_cleaner.py       # Clean and preprocess data
│   │   └── data_transformer.py   # Feature engineering and transformation
│   ├── processing/
│   │   └── data_processor.py     # Orchestrate data loading and transformation
│   ├── modeling/
│   │   ├── model_trainer.py      # Train and evaluate models
│   │   └── model_explainer.py    # SHAP-based model explainability
│   └── pipeline/
│       └── pipeline.py           # End-to-end pipeline with MLflow integration
├── config/
│   └── config.yaml               # Centralized configuration for paths and parameters
├── Dockerfile                    # Containerization for reproducible environments
├── requirements.txt              # Python dependencies
├── main.py                       # Entry point to run the pipeline
```

---

## ⚙️ Features

- 🧩 **Modular design** – Easy to extend and maintain  
- ⚙️ **Config-driven execution** – Reproducible pipelines  
- 📊 **MLflow integration** – Track experiments, parameters, metrics, and artifacts  
- 🔍 **Explainability with SHAP** – Transparent model decisions  
- 🛰 **Prefect-ready orchestration** – Scale workflows seamlessly  
- 🐳 **Dockerized environment** – Run anywhere with consistent dependencies  

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/fraud-detection-pipeline.git
cd fraud-detection-pipeline
```

### Option 1: Local installation

```bash
pip install -r requirements.txt
```

### Option 2: Run with Docker (recommended)

Build the Docker image:

```bash
docker build -t fraud-detection-pipeline .
```

Run the container:

```bash
docker run -it --rm fraud-detection-pipeline
```

---

## 🚀 Usage

1. Configure your settings in **`config/config.yaml`**  
2. Run the pipeline:

### Local

```bash
python app.py
```

### Docker

```bash
docker run -it --rm -v $(pwd)/config:/app/config fraud-detection-pipeline
```

This will:  
- Load and clean the data  
- Engineer features and transform inputs  
- Train models and evaluate performance  
- Log metrics, parameters, and artifacts to MLflow  
- Generate SHAP plots for interpretability  

---

## 📊 Supported Datasets

- **Fraud_Data.csv** → Behavioral and device-level features  
- **IpAddress_to_Country.csv** → Geolocation enrichment  
- **creditcard.csv** → Transactional features for supervised learning  

---

## 📈 Sample Output

| Model                | Dataset          | F1 Score | AUC-PR |
|----------------------|------------------|----------|--------|
| Logistic Regression  | Fraud_Data.csv   | 0.89     | 0.91   |
| LightGBM             | Fraud_Data.csv   | 0.92     | 0.94   |

Artifacts and metrics are logged in **`mlruns/`** for local MLflow tracking.

---

## 🔮 Future Enhancements

- ⚡ FastAPI endpoints for **real-time scoring**  
- 📊 Streamlit dashboard for **monitoring performance**  
- 📉 Drift detection and **automated retraining**  
- ☁️ Remote MLflow tracking with **PostgreSQL + S3**  
- 🐳 Docker Compose integration for **multi-service orchestration**  

---

## 🤝 Contributing

Contributions are welcome! 🎉  
- Fork the repo  
- Create a new branch (`feature/new-module`)  
- Submit a pull request 🚀  

---

## 📜 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

Built by **Metasebiya** — architecting **transparent, scalable, and intelligent data systems**.
