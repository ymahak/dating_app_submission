# AI / Backend Engineer Assignment - April 2026

## Task 1 - Data Preprocessing

Preprocessing is implemented in `train_model.py` directly so the same logic is reused during training and deployment.

Steps and rationale:

1. **Load and inspect**
   - Loaded `dating_dataset.csv` and validated shape, column schema, and null profile.
   - Reason: every downstream decision depends on understanding whether data is complete and consistent.

2. **Target definition**
   - Converted `match_outcome` into binary `is_good_match`.
   - Positive class: `Mutual Match`, `Date Happened`, `Relationship Formed`, `Instant Match`.
   - Reason: product decision is usually binary at serving time ("recommend or not"), and this aligns to that use case.

3. **Feature engineering**
   - `interest_count`: number of selected interests.
   - `engagement_score`: weighted combination of likes, mutual matches, and message activity.
   - `activity_score`: usage time multiplied by swipe-right ratio.
   - `profile_completeness`: normalized signal from profile pictures and bio length.
   - Reason: these are behaviorally meaningful abstractions, not just raw columns.

4. **Model-ready preprocessing**
   - Numeric columns: median imputation + scaling.
   - Categorical columns: mode imputation + one-hot encoding.
   - Reason: robust to future missing values and safe with unseen categories in production.

## Task 2 - Model Training and Evaluation

Training is in `train_model.py`.

### Why Random Forest (3-5 sentences)
I selected Random Forest because the feature set is mixed (continuous behavior metrics + categorical demographics), and the model handles this well after one-hot encoding. It captures non-linear interactions that are common in compatibility behavior, unlike linear baselines that can underfit this kind of data. It is also stable and practical for a production assignment because it performs well with limited tuning. I preferred it over more complex boosting-based approaches for this submission because interpretability, reliability, and reproducibility were more important than squeezing out small metric gains.

### Metrics used and why
- **F1 score**: balances precision and recall when both false positives and false negatives hurt user trust.
- **Precision**: helps control low-quality recommendations.
- **Recall**: helps avoid missing potentially good matches.
- **ROC-AUC**: threshold-independent signal of ranking quality.
- **Accuracy**: included for context only.

The script writes metrics to `artifacts/metrics.json`.

### Interpretation in dating-app context
- Higher precision means users see fewer low-quality match suggestions.
- Higher recall means fewer potentially compatible matches are missed.
- Strong AUC means the model is useful for ranking candidates, even before hard thresholding.
- Confidence score from `predict_proba` can be used for tiered UX (e.g., "high confidence match" badges vs standard suggestions).

## Task 3 - Live App with AI Integration

Implemented in `app.py`.

### Part A - Test model with two example profiles
Two sample predictions are auto-generated during training and saved in `artifacts/sample_predictions.json` (with predicted class and confidence).

### Part B - AI integration
The app calls Groq Chat Completions (if `GROQ_API_KEY` is present) to generate a natural-language explanation of why a profile is likely/unlikely to get a good match. If no API key is configured, the app returns a deterministic fallback explanation.

### Part C - Deploy
Run locally:

```bash
streamlit run app.py
```

Deploy to Streamlit Community Cloud and place your public URL here before submission:

`<YOUR_STREAMLIT_APP_LINK>`

## Task 4 - Research and Improvement Plan

1. **Biggest weaknesses in current pipeline**
   - Binary target compresses nuance from 10 outcomes.
   - No temporal sequence modeling (we only use snapshot features).
   - No fairness audit by sensitive demographics.
   - Feature set does not capture conversational text quality.

2. **How to improve matching with more time/resources**
   - Move from single-profile prediction to pairwise candidate scoring.
   - Add representation learning for interests and behavioral history.
   - Use calibrated gradient boosting or ranking models (pairwise/listwise objectives).
   - Add continuous online evaluation with guardrails.

3. **How to improve AI layer**
   - Ground explanations in top local feature contributions (e.g., SHAP snippets).
   - Add safety policy filters for generated language.
   - Personalize explanation style by user preference.
   - Log and evaluate explanation helpfulness feedback.

4. **One thing I would do differently from scratch**
   - I would start with a pair-construction and ranking objective from day one, since dating recommendations are fundamentally ranking problems rather than only binary classification problems.
