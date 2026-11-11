# TTF spread research

The notebooks use the saved raw CSVs. No Bloomberg, GIE or weather download is run.

Run from this project folder with the Python environment listed in `requirements.txt`:

```powershell
python run_project.py
python validate_project.py
```

Or run the notebooks in this order: `2. Data Processing`, `3. Final_DB/final_table.ipynb`, `4. Modelling/spread_models.ipynb`, `4. Strategy/spread_strategy.ipynb`, `5. Backtest/spread_backtest.ipynb`.

`run_project.py` saves executed notebook outputs and a source/dependency manifest. It accepts `--section pipeline`, `models`, `strategy` or `backtest` to repeat one stage.

The primary research horizon is five trading sessions, with one and ten sessions as diagnostics. Signals use information at date t, enter at the next session's settlement and exit h sessions after entry. The contract pair must remain unchanged. The 2024 onward evaluation settings were fixed before this implementation's results were examined; prior human use of this period is unknown.

Read `Documentation/Project Change Log.html` for the changes, results and remaining limits. Historical documentation remains as the original design record and is superseded where the change log says otherwise.

These are research results. Bloomberg generic contract mappings are inferred, fundamentals contain later revisions, costs are assumptions, and equal-MWh positions can require fractional MW. No historical execution or margin-capital claim is made.
