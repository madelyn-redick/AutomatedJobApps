# Personal Job Application Matcher
## What It Does

1. Pulls fresh postings from Adzuna (job listing aggregator, ~1,000 free API calls/month)
2. Checks against a local SQLite database to avoid duplicates
3. Scores every posting against resume and cover letter using TF-IDF cosine similarity
4. Flags postings containing priority keywords
5. Writes new matches above threshold to `matched_jobs.csv`.

## Setup

1. **Install dependencies** (one time):
   ```
   pip install -r requirements.txt
   ```

2. **Get a free Adzuna key**: register at https://developer.adzuna.com/

3. **Copy the env template and fill it in**:
   ```
   cp .env.example .env
   ```
   Then edit `.env`:
   - Add your Adzuna `app_id` and `app_key`
   - Set `SEARCH_KEYWORDS` and `SEARCH_LOCATION`
   - Set `PRIORITY_KEYWORDS` to whatever language should get flagged

4. **Paste resume and cover letter** into
   `resume_and_cover_letter.txt` (plain text, no formatting needed)

5. **Run it**:
   ```
   python job_scanner.py
   ```

You'll see a summary printed, and new matches appended to
`matched_jobs.csv`. All postings (matched or not) are stored in
`jobs.db` so re-runs only report new postings.
