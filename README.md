# Client Attrition Risk Prediction (ML + Flask API)

This project builds and deploys a machine learning pipeline that helps identify corporate clients at risk of attrition. The solution enables a small client management team to focus their efforts on high-risk clients by using an ML-powered risk scoring system, deployed and served through a Flask API.

**GitHub Repository:**  
[github.com/LauRacoon2110/udc_mldevops_cp4_dynamic_risk_assessment](https://github.com/LauRacoon2110/udc_mldevops_cp4_dynamic_risk_assessment)

---

## Project Structure

```
.
├── apicalls.py
├── app.py
├── config.json
├── cronjob.txt
├── deployment.py
├── diagnostics.py
├── fullprocess.py
├── ingestion.py
├── LICENSE
├── mypy.ini
├── pyproject.toml
├── README.md
├── reporting.py
├── requirements.txt
├── scoring.py
├── training.py
├── utils.py
├── wsgi.py
├── .python-version
├── .pre-commit-config.yaml
├── .gitignore
├── ingesteddata/
│   ├── finaldata.csv
│   └── ingestedfiles.txt
├── models/
│   └── .gitkeep
├── practicedata/
│   ├── dataset1.csv
│   └── dataset2.csv
├── practicemodels/
│   ├── apireturns.txt
│   ├── confusionmatrix.png
│   ├── latestscore.txt
│   └── trainedmodel.pkl
├── production_deployment/
│   ├── ingestedfiles.txt
│   ├── latestscore.txt
│   └── trainedmodel.pkl
├── screenshots/
│   └── fullprocess_run_output_prod.png
├── sourcedata/
│   ├── dataset3.csv
│   └── dataset4.csv
├── testdata/
│   └── testdata.csv
```

---

## Pipeline Steps

This project uses a modular ML pipeline:

1. **Data Ingestion** (`ingestion.py`)  
   Merges new CSVs into a single file (`finaldata.csv`) for model training.

2. **Model Training** (`training.py`)  
   Trains a `LogisticRegression` model using selected features and target.

3. **Model Scoring** (`scoring.py`)  
   Evaluates the trained model using the F1 score.

4. **Model Deployment** (`deployment.py`)  
   Copies model artifacts and logs into the production folder.

5. **Diagnostics** (`diagnostics.py`)  
   Performs:
   - Summary stats
   - Missing value check
   - Execution time tracking
   - Outdated packages scan

6. **Reporting** (`reporting.py`)  
   Creates and saves a confusion matrix plot.

7. **API** (`app.py`)  
   Flask server for exposing prediction and diagnostic endpoints.

8. **Pipeline Automation** (`fullprocess.py`)  
   Executes the entire flow, checks for new data, handles model drift, redeploys models, and triggers reporting and API calls.

---

## Model Details

- **Model**: `LogisticRegression` (scikit-learn)
- **Features**:
  - `lastmonth_activity`
  - `lastyear_activity`
  - `number_of_employees`
- **Target**: `exited` (binary classification)
- **Metric**: `F1 Score`

---

## 🔌 API Endpoints

| Endpoint         | Method | Description                                       |
|------------------|--------|---------------------------------------------------|
| `/prediction`     | POST   | Predict churn for a given CSV file               |
| `/scoring`        | GET    | Return the model's F1 score                      |
| `/summarystats`   | GET    | Get data mean, median, and standard deviation    |
| `/diagnostics`    | GET    | Check execution time, nulls, outdated packages   |

---

## Manual Execution

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Production Mode

Before running the a single pipeline step, make sure to check the environment settings `config.json`:
e.g. 

```json
"active_environment": "dev"
```

This ensures the pipeline uses `/sourcedata/`, `/models/`, and `/production_deployment/`.

### 3. Run each pipeline step manually (in dev or prod mode)

```bash
python ingestion.py
python training.py
python scoring.py
python deployment.py
python diagnostics.py
python reporting.py
```

### 4. Run the Flask API

```bash
python app.py
```

Then test it using:

```bash
python apicalls.py
```

---

## Run the Full Pipeline

You can run everything with:

```bash
python fullprocess.py
```

This script:
- Detects new data
- Ingests it
- Checks for model drift
- Retrains and redeploys if needed
- Starts the API and hits endpoints
- Generates a new confusion matrix

---

## Automate with Cron

The pipeline can be triggered every 10 minutes using the following cron job (`cronjob.txt`):

```cron
*/10 * * * * /home/laura/projects/udc_mldevops_cp4_dynamic_risk_assessment/.venv/bin/python /home/laura/projects/udc_mldevops_cp4_dynamic_risk_assessment/fullprocess.py >> /home/laura/projects/udc_mldevops_cp4_dynamic_risk_assessment/fullprocess.log 2>&1
```

---

## Linting & Formatting

Pre-commit tools used:

- `flake8`
- `black`
- `mypy`
- `isort`
- `pytest`

Enable with:

```bash
pre-commit install
pre-commit run --all-files
```

---

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**  
See [`LICENSE`](./LICENSE) for more information.
