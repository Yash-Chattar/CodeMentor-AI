# CodeMentor AI — Accuracy-focused update

## What changed
- Classification now predicts next week's Weak/Average/Strong category, rather than classifying the current completed week's category.
- Features include current accuracy, previous accuracy, rolling history, improvement, attempts, solved count, hint count, time, solved rate and hint rate.
- Target rows are retained only when the next observation is the immediately following week.
- Model selection uses a chronological validation period; the final test period is not used for selection.
- Compared Logistic Regression, Random Forest, balanced Random Forest, Extra Trees, Gradient Boosting and ANN (MLP).
- Regression models forecast next week's accuracy and are compared with a persistence baseline.

## Setup with uv
```powershell
uv sync
uv run streamlit run app.py
```

Or manual venv activation:
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## Important
The bundled data is synthetic. The updated next-week classification test accuracy is about 82.4% on this demo dataset. It is not 90–95%, and this must not be represented as real-world performance. The old same-week classification metric is not directly comparable because the target task has changed.

See `reports/updated_model_evaluation.md` for the detailed evaluation.
