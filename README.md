# CampusPath — Student Placement Predictor

This project converts the research notebook into a VS Code-friendly Python project and adds a React frontend with a FastAPI backend.

## Included
- Original research experiment pipeline: five ML algorithms × All / Top-10 / Top-7 / Top-5 / Top-3 feature sets.
- SHAP feature ranking and research result exports.
- Training script that saves model pipelines for the web app.
- FastAPI endpoints for schema, model options, global SHAP ranking, saved-model analytics and prediction.
- React + Vite frontend with model comparison charts, held-out metrics, confusion matrices, ROC curves, dataset class distribution, student prediction and SHAP explainability.
- Separate client-side pages at `/`, `/prediction`, `/comparison`, `/explainable-ai` and `/about` with browser history support; each route renders only its own content.
- Installable PWA shell with standalone display mode; prediction still needs the FastAPI service.
- The dataset embedded in the supplied notebook is included at `data/campus_placement_data.csv`.

## Important notes
- The app is a research/educational prototype, not a hiring decision system.
- Prediction probability is a model estimate, not a guarantee of employment.
- The SHAP chart is **global** feature importance from training data. It is not an individual-level explanation for the current prediction.
- The notebook's research workflow is retained. For a rigorous paper, review the feature-selection validation design: the original script ranks features once using the full training split before cross-validation. For unbiased CV estimates, SHAP feature ranking should be recomputed inside each CV training fold.
- The saved app models are separate fitted pipelines; run the app training command before starting the frontend.
- Use a small sample first to verify your local environment. Train on the full dataset after confirming everything works.

## Feature subsets in plain language
The training script calculates SHAP feature importance on the training split, ranks features by mean absolute SHAP value, and selects the top 3, 5, 7, or 10 features for those configurations. The `All` set uses every model input column in the original dataset order. It saves a separate fitted model pipeline for each algorithm and subset. During prediction, the API reads the feature list from `metadata.json` and loads that exact model artifact; it preserves that model's feature ordering.

Mean absolute SHAP measures how much a feature influences model output on average, regardless of whether the influence pushes a prediction up or down. It describes overall model behaviour and does not establish causation or explain one student's prediction.

## Generate model analytics without retraining
The evaluation output was not part of the original saved artifacts. Generate it from the existing saved pipelines using the same cleaned data and stratified 80/20 split (`random_state=42`) used to train those app pipelines:
```powershell
python -m backend.app.evaluate_saved_models
```
This loads all 25 saved models, evaluates them on the held-out 20% split at a 0.5 threshold, and writes `backend/results/saved_model_metrics.csv` plus `saved_model_analytics.json`. The JSON contains dataset class counts, all five metrics, 2×2 confusion matrices and downsampled ROC curves. It does not fit or overwrite any model. The dashboard `/api/analytics` endpoint uses this output. Re-run the evaluator whenever the dataset or saved model artifacts change.

The visual analytics are populated from `GET /api/analytics`; global SHAP magnitudes are served by `GET /api/feature-ranking`; model and feature order metadata come from `GET /api/models`; and prediction posts to `POST /api/predict`. If evaluation or SHAP outputs are missing, the UI shows an unavailable state rather than substituting illustrative values. The Overview class-distribution chart shows the dataset target distribution, not the probability for an individual student.

## Dashboard pages and metric consistency

- `/` — overview KPIs and dataset placement distribution, with a compact F1 comparison.
- `/prediction` — student profile form, saved model/subset selectors and returned prediction probability.
- `/comparison` — metric, feature subset, ROC and confusion matrix views, backed by saved analytics JSON.
- `/explainable-ai` — global SHAP ranking, feature subset lists and the generated beeswarm image when available.
- `/about` — project, dataset, preprocessing, evaluation and system workflow.

All model evaluations use the same 20,000-row stratified test split. At a 0.5 threshold, Accuracy, Precision, Recall and F1 agree with each corresponding confusion matrix. ROC-AUC is computed from held-out positive-class probabilities; the saved display curves are downsampled, so recomputing AUC from their points can differ slightly. Feature selection is ranked once on the training split before cross-validation; this avoids held-out test leakage, but fold-specific SHAP ranking would be needed for unbiased cross-validation estimates.

## 1. Open the project in VS Code
1. Extract this ZIP file to a folder, for example `Documents/Student_Placement_Prediction`.
2. Open VS Code.
3. Choose **File → Open Folder** and select the extracted `Student_Placement_Prediction` folder.
4. Install the VS Code extensions **Python** (Microsoft) and **Pylance**.

## 2. Set up Python (Windows PowerShell terminal)
Open **Terminal → New Terminal** in VS Code and run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt terminal and run:
```cmd
.venv\Scripts\activate.bat
```

## 3. Run the research experiment code
Quick smoke test:
```powershell
python experiments/run_experiments.py --data data/campus_placement_data.csv --out results_quick --sample 10000 --folds 3
```

Full experiment:
```powershell
python experiments/run_experiments.py --data data/campus_placement_data.csv --out results --folds 5
```

This can take a long time because it runs SHAP plus 25 model configurations. The app model training below is a separate step.

## 4. Train models for the frontend
Start with a 10,000-row sample:
```powershell
python -m backend.app.train_models --sample 10000
```

When this completes, it creates model files and `metadata.json` under `backend/artifacts/`.
For final local models, omit `--sample`:
```powershell
python -m backend.app.train_models
```

## 5. Start the backend
Open a **second VS Code terminal**, from the project root, activate `.venv` if needed, then run:
```powershell
uvicorn backend.app.main:app --reload
```

Check:
- API health: http://127.0.0.1:8000/api/health
- Interactive API docs: http://127.0.0.1:8000/docs

Keep this terminal running.

## 6. Start the frontend
Open a **third VS Code terminal**:
```powershell
cd frontend
npm install
npm run dev
```

Open the local URL Vite prints (usually http://localhost:5173).

The Vite development server serves direct page routes and preserves them on refresh. Use the app navigation or browser back/forward controls to move between pages.

## 7. Install on a phone
Deploy the Vite production build (`frontend/dist`) over HTTPS and keep the FastAPI API reachable from the phone. Configure `VITE_API_URL` to the deployed API origin before building and set backend `CORS_ORIGINS` to the frontend origin (comma-separated if needed). The frontend default is `http://127.0.0.1:8000`, which is only suitable for local development on the same computer. HTTPS is required for service workers and installation outside localhost.

- **Android / Chrome:** open the HTTPS app URL, then choose **Install app** in CampusPath or Chrome's menu.
- **iPhone / Safari:** open the HTTPS app URL, tap **Share**, then **Add to Home Screen**. iOS does not expose the same browser install prompt.

The service worker caches only the app shell (HTML/manifest/icon). It does not cache student inputs, prediction requests, or prediction responses. A working backend connection is required to generate predictions; this is not a fully offline prediction app.

## Project structure
```text
backend/app/             FastAPI API and model-training script
backend/artifacts/       Existing trained pipelines, model metadata and feature order
backend/results/         Numeric SHAP ranking output
data/                    Placement dataset
experiments/             Research and evaluation pipeline
frontend/src/            React UI and responsive styles
frontend/public/         PWA manifest, icon and shell service worker
requirements.txt         Python dependencies
```

## Troubleshooting
- **Backend unavailable:** start `uvicorn backend.app.main:app --reload` in the project root; check `http://127.0.0.1:8000/api/health`.
- **SHAP ranking unavailable:** ensure `backend/results/shap_feature_ranking.csv` exists and contains `feature` and `mean_abs_shap` columns. The prediction page reads its numeric SHAP magnitudes and joins them to saved feature order.
- **Analytics dashboard unavailable:** run `python -m backend.app.evaluate_saved_models` after the saved pipelines are present.
- **Phone cannot predict:** `127.0.0.1` on a phone refers to the phone. Set `VITE_API_URL` to a reachable HTTPS backend, rebuild, and deploy both services.
- **Install option missing:** use a supported browser on HTTPS (or localhost during development); on iPhone use Safari's **Share → Add to Home Screen**.

## 8. Team workflow with GitHub
- Create one GitHub repository and upload this project.
- Each member clones the repository to their own laptop.
- Work on separate branches and separate files/modules.
- Commit and push changes; use pull requests or merge carefully.
- Do not commit `.venv`, `node_modules`, generated model `.joblib` files or large result folders.
- Share the dataset through your agreed team storage if GitHub's file-size limits are an issue.

- `No module named ...`: activate `.venv` and run `pip install -r requirements.txt`.
- API says models are not trained: run `python -m backend.app.train_models --sample 10000`.
- Port already in use: stop the other process or change the port.
- If package installation fails on an older Python version, install Python 3.10 or 3.11 and recreate `.venv`.
