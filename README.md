# Retail Customer Sales EDA

## Start the app

The source CSV is in this folder. On Windows, double-click `run_app.bat`. From PowerShell, use the workspace environment:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m streamlit run Project1_EDA.py
```

If you copied this folder outside the workspace, create and activate a virtual environment here, install `requirements.txt`, then run `python -m streamlit run Project1_EDA.py`. Running `Project1_EDA.py` directly from VS Code also forwards to Streamlit.

## Project files

- `Project1_EDA.py` is the interactive app. It includes region/date filters, quality checks, numeric summaries, distributions, grouped business questions, and CSV downloads.
- `Project1_Exploratory_Data_Analysis.ipynb` records the same analysis in notebook form.
- `retail_sales_cleaned.csv` contains the 7,023 deduplicated analysis rows with flags and engineered fields.
- `data_quality_audit.csv` and `duplicate_rows.csv` provide the source missing-value audit and the 61 exact duplicate rows.
- `EDA_Insights_Report.pdf` summarizes findings from this CSV in plain business language.
- `Hypotheses_and_Test_Cases.pdf` lists follow-up hypotheses and the data checks used in the project.

The source file remains unchanged. Missing values stay blank; invalid values are flagged; unusual but possible values are retained for review.
