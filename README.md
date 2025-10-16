AI-Based Predictive House Price Estimation using Deep Learning and Ensemble Machine Learning Models

Streamlit UI entrypoint: `app.py` — run `streamlit run app.py`

Project scaffold for: "AI-Based Predictive House Price Estimation using Deep Learning on Kaggles Ames Housing Dataset".

Quick start

1. Place your dataset CSV in `data/data.csv` (or `data/train.csv`).
2. Create a Python virtual environment and install dependencies:

   python -m venv .venv
   .\\.venv\\Scripts\\Activate.ps1
   pip install -r requirements.txt

3. Inspect the data (optional):

   python src/inspect_data.py --file data/data.csv

4. Run a baseline training (optional - archived in `archive/`):

   python src/train_baseline.py --file data/data.csv

Files
- `app.py`: Streamlit web UI (main entrypoint).
- `data/`: expected CSV dataset.
- `requirements.txt`: primary dependencies for running the Streamlit app.
- `Dockerfile`, `docker-compose.yml`: optional Docker deployment for Streamlit.

Notes
- The Kaggle Ames dataset is small (~1460 rows). Use cross-validation and regularization for deep learning models.
