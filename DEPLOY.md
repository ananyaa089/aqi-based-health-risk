# Put the dashboard online (Streamlit Community Cloud, free)

You need a free GitHub account and a free Streamlit Community Cloud account (sign in with GitHub).

1. Finish the pipeline locally first (`python scripts/run_all.py`), so `data/clean/dashboard`, `data/clean/report` and the
   three small results CSVs exist. The `.gitignore` already keeps raw data and the 60 MB hourly file OUT of the repo.
2. In the project folder:
       git init
       git add .
       git status          # check: data/clean/dashboard + report + 3 results CSVs are listed; raw data and master_aqi_hourly.csv are not
       git commit -m "Delhi AQI early-warning capstone"
3. On github.com create an empty repository (public is simplest), then:
       git remote add origin https://github.com/<your-username>/<repo>.git
       git branch -M main
       git push -u origin main
4. Go to share.streamlit.io -> New app -> pick the repository, branch `main`, main file path `app/dashboard.py` -> Deploy.
   Streamlit uses `app/requirements.txt` (dashboard-only, installs in about a minute).
5. When it opens, the sidebar "Data folder" is already `data/clean`, which exists in the repo, so it just works.
6. After re-running the models, redo steps 2-3 (`git add`, `git commit`, `git push`); the site redeploys automatically.

Notes
* The dashboard shows replay-mode forecasts only (no live feed). Say so in your demo.
* Do not commit the raw OpenCity/CPCB files or anything you are not allowed to redistribute; read each dataset's licence.
* Public repo = anyone can see your code and results. Use a private repo if your course prefers it
  (Streamlit Community Cloud supports private repos once you grant access).
