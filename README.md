# Dating App Match Predictor 

## Files

- `train_model.py`: End-to-end preprocessing, feature engineering, model training, and evaluation.
- `app.py`: Streamlit app for interactive prediction + AI explanation layer.
- `ASSIGNMENT_REPORT.md`: Written answers for Tasks 1-4.
- `dating_dataset.csv`: Raw dataset.
- `artifacts/`: Generated model and evaluation outputs:
  - `match_pipeline.pkl`
  - `feature_columns.pkl`
  - `metrics.json`
  - `sample_predictions.json`

## Run Locally

1) Install dependencies:

```bash
pip install -r requirements.txt
```

2) Train and generate artifacts:

```bash
python train_model.py
```

3) Start the app:

```bash
streamlit run app.py
```

## AI Integration Setup (optional)

- Set `GROQ_API_KEY` in your environment to enable LLM-generated explanations.
- Without the key, the app still works and shows a deterministic fallback explanation.
