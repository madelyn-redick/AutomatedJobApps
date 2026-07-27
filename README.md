# Personal Job Application Machine -- Phase 1 (Scan, Score, Log)

This covers steps 1, 2, 5, and 7 from your project outline: continuously
scanning job sources, matching/scoring against your resume, highlighting
priority keywords, and logging results. Everything here is free.

## What it does

1. Pulls fresh postings from three free sources:
   - **Adzuna** (aggregator, ~1,000 free API calls/month)
   - **Greenhouse** job boards (free, no key -- per company)
   - **Lever** job boards (free, no key -- per company)
2. Deduplicates against a local SQLite database so you never see the same
   posting twice across runs.
3. Scores every posting against your resume + cover letter using TF-IDF
   cosine similarity (runs locally, no paid embedding API).
4. Flags postings containing any of your "priority keywords."
5. Writes new matches above your threshold to `matched_jobs.csv`.

## Setup

1. **Install dependencies** (one time):
   ```bash
   pip install -r requirements.txt
   ```

2. **Get a free Adzuna key**: register at https://developer.adzuna.com/
   -- it's instant and free, gives you an `app_id` and `app_key`.

3. **Copy the env template and fill it in**:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env`:
   - Add your Adzuna `app_id` / `app_key`
   - Set `SEARCH_KEYWORDS` and `SEARCH_LOCATION`
   - Add `GREENHOUSE_COMPANIES` / `LEVER_COMPANIES` -- any companies you
     want to watch directly. Find their board token from their careers
     page URL:
     - Greenhouse: `boards.greenhouse.io/{token}`
     - Lever: `jobs.lever.co/{token}`
   - Set `PRIORITY_KEYWORDS` to whatever language should get flagged
     (e.g. "LLM, generative AI, remote")

4. **Paste your resume + cover letter** into
   `resume_and_cover_letter.txt` (plain text, no formatting needed).

5. **Run it**:
   ```bash
   python job_scanner.py
   ```

You'll see a summary printed, and new matches appended to
`matched_jobs.csv`. All postings (matched or not) are stored in
`jobs.db` so re-runs only report *new* postings.

## Running it automatically

**Mac/Linux (cron):** run `crontab -e` and add, e.g., once a day at 8am:
```
0 8 * * * cd /path/to/job_scanner && /usr/bin/python3 job_scanner.py >> run.log 2>&1
```

**Windows:** use Task Scheduler to run `job_scanner.py` on a daily trigger.

**Free cloud option:** a GitHub Actions workflow on a `schedule` cron
trigger can run this for free (public repos get free minutes; private
repos get a free monthly quota). Ask if you want that workflow file --
it's a natural next addition once this works locally.

## About the Adzuna quota

The free tier is ~1,000 calls/month (about 33/day). Each page of
results here is one call, and `ADZUNA_MAX_PAGES` controls how many
pages you pull per run. Running this **once a day** with 1-2 pages
keeps you comfortably inside the free quota even with daily runs.
Greenhouse and Lever calls are free and don't count against this.

## Getting matches into Google Sheets

This phase writes to a CSV, which you can:
- Manually import into Sheets (File > Import), or
- Auto-sync for free using a Google Cloud service account + the
  `gspread` library (no billing required for this use case) -- this is
  a natural Phase 2 addition, since it's a self-contained piece
  (matches your outline's step 7).

## What's intentionally NOT included yet

- **Auto-submitting Easy Apply applications** (step 3) and **autofill
  for other ATS platforms** (step 6) both involve browser automation
  against sites whose Terms of Service prohibit automated applying,
  and LinkedIn in particular detects and bans this kind of automation.
  I'd suggest building those as **semi-automated** (the tool pre-fills,
  you click submit) rather than fully automatic, to keep your accounts
  safe. Happy to build that phase next if you want to go there.
