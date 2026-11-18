# AgriNova — Streamlit Community Cloud deployment

This deployment runs the Streamlit interface and the trained ML model in the same Python process. It does not start a separate FastAPI server, so no Render service or paid compute card is required for this version.

## Repository layout

Keep these files and folders at the repository root:

- `app.py`
- `requirements.txt`
- `artifacts/model.joblib`
- `artifacts/metadata.json`
- `src/__init__.py`
- `src/features.py`
- `src/predictor.py`

The original `src/api.py` is retained for a future separate API deployment, but this Streamlit deployment does not call it.

## Deploy

1. Copy the contents of this project folder into the root of the GitHub repository `adilharun19-create/AgriNova` (do not create an extra nested `AgriNova-main` folder in the repository).
2. Commit and push the updated files to branch `main`.
3. Open Streamlit Community Cloud and create an app with:
   - Repository: `adilharun19-create/AgriNova`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: choose any available subdomain.
4. In Advanced settings, select the same Python minor version used to train the model if it is available. The supplied model metadata records Python `3.15.2` and scikit-learn `1.9.1`.
5. Deploy. The first build can take several minutes while dependencies are installed.

## Important

The `requirements.txt` pins NumPy, pandas, scikit-learn, and joblib to the versions recorded in `artifacts/metadata.json`. Do not loosen these versions without retraining or validating the serialized model. If the Python `3.15` option is not available in the deployment selector, do not guess a different version; check the deployment build log and validate compatibility first.
