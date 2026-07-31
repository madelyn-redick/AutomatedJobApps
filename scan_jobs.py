### JOB APPLICATION MATCHER
# Madelyn Redick
# July 2026

import os
from click import launch
from dotenv import load_dotenv
import csv
import sqlite3
import hashlib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import pandas as pd
from bs4 import BeautifulSoup
import html
from dash import Dash, html as dash_html, dcc, Input, Output, State, no_update, clientside_callback
import dash_ag_grid as dag

load_dotenv()


# config
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "")
ADZUNA_COUNTRY = os.getenv("ADZUNA_COUNTRY", "us")

SEARCH_KEYWORDS = os.getenv("SEARCH_KEYWORDS", "data scientist")
SEARCH_LOCATION = os.getenv("SEARCH_LOCATION", "")
MAX_DAYS_OLD = int(os.getenv("MAX_DAYS_OLD", "3"))
RESULTS_PER_PAGE = int(os.getenv("ADZUNA_RESULTS_PER_PAGE", "20"))
MAX_PAGES = int(os.getenv("ADZUNA_MAX_PAGES", "2"))
GREENHOUSE_MAX_DAYS_OLD = 14

GREENHOUSE_COMPANIES = [
    c.strip() for c in os.getenv("GREENHOUSE_COMPANIES", "").split(",") if c.strip()
]

RESUME_FILE = os.getenv("RESUME_FILE", "resume_and_cover_letter.txt")
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.15"))
PRIORITY_KEYWORDS = [k.strip().lower() for k in os.getenv("PRIORITY_KEYWORDS", "").split(",") if k.strip()]
EXCLUDE_KEYWORDS = [k.strip().lower() for k in os.getenv("EXCLUDE_KEYWORDS", "").split(",") if k.strip()]

OUTPUT_CSV = os.getenv("OUTPUT_CSV", "matched_jobs.csv")
DB_FILE = os.getenv("DB_FILE", "jobs.db")
APPLIED_CSV = os.getenv("APPLIED_CSV", "applied_jobs.csv")
IGNORED_CSV = os.getenv("IGNORED_CSV", "ignored_jobs.csv")

def parse_weight_dict(value):
    """ convert comma-separated environment variable string into a dictionary. example: "python:5,sql:3" becomes: {"python": 5, "sql": 3}

    Args:
        value (str): raw environment variable string

    Returns:
        dict: dictionary of terms and their numeric weights
    """

    weights = {}
    if not value:
        return weights

    for item in value.split(","):
        try:
            keyword, weight = item.split(":")
            weights[keyword.strip().lower()] = float(weight)

        except ValueError:
            print(f"[warning] Invalid weight format: {item}")

    return weights

SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.6"))
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.3"))
PRIORITY_WEIGHT = float(os.getenv("PRIORITY_WEIGHT", "0.1"))

MATCH_WEIGHTS = parse_weight_dict(os.getenv("MATCH_WEIGHTS", ""))
PENALTY_WEIGHTS = parse_weight_dict(os.getenv("PENALTY_WEIGHTS", ""))

def clean_html_text(text):
    """ convert HTML job descriptions into clean plain text

    Returns:
        str: cleaned description with no empty lines
    """

    if not text:
        return ""

    # decode HTML entities
    text = html.unescape(text)

    # remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(" ")

    # remove leftover whitespace
    text = " ".join(text.split())

    return text

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_hash TEXT PRIMARY KEY,
            title TEXT,
            company TEXT,
            location TEXT,
            description TEXT,
            url TEXT,
            source TEXT,
            posted_date TEXT,
            scraped_date TEXT,
            keyword_score REAL,
            semantic_score REAL,
            penalty_score REAL,
            final_score REAL,
            score REAL,
            is_priority INTEGER,
            status TEXT DEFAULT 'new'
        )
        """
    )
    conn.commit()
    return conn

def job_hash(title, company, posted_date):
    # normalize job title, company, and posted date
    raw = f"{title.strip().lower()}|{company.strip().lower()}|{posted_date}"

    # create SHA-256 hash representation to save as primary key
    return hashlib.sha256(raw.encode()).hexdigest()

def get_adzuna_jobs():
    """ retrieves job listings that match search criteria

    Returns:
        jobs (list[dict]): list of job postings, each dictionary contains job title, company, location, description, application URL, source, and posted date
    """
    print("Scanning Adzuna:")
    jobs = []

    # loop through each page of API results until page limit reached or no more jobs
    for page in range(1, MAX_PAGES + 1):
        url = f"https://api.adzuna.com/v1/api/jobs/{ADZUNA_COUNTRY}/search/{page}"
        params = {
            "app_id": ADZUNA_APP_ID,
            "app_key": ADZUNA_APP_KEY,
            "what": SEARCH_KEYWORDS,
            "where": SEARCH_LOCATION,
            "max_days_old": MAX_DAYS_OLD,
            "results_per_page": RESULTS_PER_PAGE,
            "sort_by": "date",
        }

        # send request to Adzuna API, parse JSON response
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  [error] Adzuna page {page}: {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        # format
        for r in results:
            jobs.append(
                {
                    "title": r.get("title", "").strip(),
                    "company": (r.get("company") or {}).get("display_name", "Unknown"),
                    "location": (r.get("location") or {}).get("display_name", ""),
                    "description": clean_html_text(r.get("description", "")),
                    "url": r.get("redirect_url", ""),
                    "source": "adzuna",
                    "posted_date": r.get("created", "")[:10],
                }
            )
    print(f"Adzuna fetched {len(jobs)} job postings")
    return jobs

def get_greenhouse_jobs():
    """ retrieves job listings that match search criteria from public Greenhouse Jobs API

        Returns:
            jobs (list[dict]): list of job postings, each dictionary contains job title, company, location, description, application URL, source, and posted date
    """
    print("Scanning Greenhouse:")
    jobs = []

    # loop through each company configured to use Greenhouse
    for company in GREENHOUSE_COMPANIES:
        url = f"https://boards-api.greenhouse.io/v1/boards/{company}/jobs"

        # send request to company's Greenhouse job board
        try:
            resp = requests.get(url, params={"content": "true"}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            # skip company if request fails, continue
            print(f"  [error] Greenhouse '{company}': {e}")
            continue

        # format
        for r in data.get("jobs", []):

            job_url = r.get("absolute_url", "")

            # only keep Greenhouse URLs
            if "greenhouse" not in job_url.lower():
                continue

            posted_date = (r.get("updated_at") or "")[:10]

            # skip jobs without a valid date
            if not posted_date:
                continue

            try:
                posted_datetime = datetime.strptime(posted_date, "%Y-%m-%d")
            except ValueError:
                continue

            # only keep jobs posted within the last 14 days
            if datetime.now() - posted_datetime > timedelta(days=GREENHOUSE_MAX_DAYS_OLD):
                continue

            jobs.append(
                {
                    "title": r.get("title", "").strip(),
                    "company": company,
                    "location": (r.get("location") or {}).get("name", ""),
                    "description": clean_html_text(r.get("content", "")),
                    "url": job_url,
                    "source": "greenhouse",
                    "posted_date": posted_date,
                }
            )

    print(f"Greenhouse: fetched {len(jobs)} postings across "f"{len(GREENHOUSE_COMPANIES)} companies")

    return jobs

def load_resume_text():
    with open(RESUME_FILE, "r", encoding="utf-8") as f:
        return f.read()

def filter_excluded_jobs(jobs):
    """ remove jobs containing words should never be considered
    """

    filtered_jobs = []
    removed_count = 0

    for job in jobs:
        text = (
            job["title"] + " " +
            job["description"]
        ).lower()

        if any(keyword in text for keyword in EXCLUDE_KEYWORDS):
            removed_count += 1
            continue

        filtered_jobs.append(job)

    print(f"Removed {removed_count} jobs due to exclusion keywords")

    return filtered_jobs

def score_jobs(jobs, resume_text):
    """ score job postings using a hybrid matching algorithm
        combines:
            1. TF-IDF cosine similarity between resume and job description
            2. Weighted keyword matching for desired skills
            3. Penalties for undesirable job characteristics
            4. Priority keyword bonuses

    Args:
        jobs (list[dict]): job postings to score
        resume_text (str): resume and cover letter text

    Returns:
        jobs (list[dict]): jobs with added scoring fields:
            - semantic_score
            - keyword_score
            - penalty_score
            - final_score
            - is_priority
    """

    if not jobs:
        return jobs

    # build text corpus containing resume and all job descriptions
    descriptions = [(j["title"] + " " + j["description"]) for j in jobs]
    corpus = [resume_text] + descriptions

    # convert text into TF-IDF vectors
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    # convert corpus into TF-IDF vectors, remove stop words
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vectorizer.fit_transform(corpus)

    # separate resume vector from job vectors
    resume_vec = tfidf[0:1]
    job_vecs = tfidf[1:]

    # compute cosine similarity between resume and each job description
    semantic_scores = cosine_similarity(resume_vec, job_vecs).flatten()

    # keyword scoring
    for job, semantic_score in zip(jobs, semantic_scores):

        text = (job["title"] + " " + job["description"]).lower()
        keyword_score = 0
        penalty_score = 0

        # boost desired skills
        for keyword, weight in MATCH_WEIGHTS.items():
            if keyword in text:
                keyword_score += weight

        # penalize undesirable requirements
        for keyword, penalty in PENALTY_WEIGHTS.items():
            if keyword in text:
                penalty_score += penalty

        # check if job contains priority terms
        is_priority = any(
            keyword in text
            for keyword in PRIORITY_KEYWORDS
        )

        # normalize
        normalized_keyword_score = min(keyword_score / 25, 1)
        normalized_penalty_score = max(penalty_score / 20, -2)

        priority_bonus = 1 if is_priority else 0

        # final weighted score
        final_score = (semantic_score * SEMANTIC_WEIGHT + normalized_keyword_score * KEYWORD_WEIGHT + priority_bonus * PRIORITY_WEIGHT + normalized_penalty_score * 0.2)

        job["semantic_score"] = round(float(semantic_score), 4)
        job["keyword_score"] = keyword_score
        job["penalty_score"] = penalty_score
        job["final_score"] = round(float(final_score), 4)
        job["score"] = job["final_score"]
        job["is_priority"] = is_priority

    return jobs

def save_new_jobs(conn, jobs):
    """ save jobs to database. checks whether each job already exists using its hash, new jobs are inserted with their scoring breakdown, jobs that exceed the match threshold or contain priority, keywords are returned for CSV export

    Args:
        conn (sqlite3.Connection): active database connection
        jobs (list[dict]): scored job postings

    Returns:
        new_matches (list[dict]): new high-value job matches
    """

    new_matches = []

    cur = conn.cursor()

    for j in jobs:

        # create unique ID for job posting
        h = job_hash(j["title"], j["company"], j["posted_date"])
        j["job_hash"] = h

        # skip jobs already stored
        cur.execute(
            "SELECT 1 FROM jobs WHERE job_hash = ?",
            (h,)
        )

        if cur.fetchone():
            continue

        # insert new job
        cur.execute(
            """
            INSERT INTO jobs
            (
                job_hash,
                title,
                company,
                location,
                description,
                url,
                source,
                posted_date,
                scraped_date,
                semantic_score,
                keyword_score,
                penalty_score,
                score,
                is_priority,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new')
            """,
            (
                h,
                j["title"],
                j["company"],
                j["location"],
                j["description"],
                j["url"],
                j["source"],
                j["posted_date"],

                datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                ),

                j["semantic_score"],
                j["keyword_score"],
                j["penalty_score"],
                j["score"],
                int(j["is_priority"]),
            ),
        )

        # keep only jobs worth reviewing
        if (
            j["score"] >= MATCH_THRESHOLD
            or j["is_priority"]
        ):
            new_matches.append(j)


    conn.commit()
    return new_matches

def write_matches_csv(matches):
    headers = ["job_hash", "title", "company", "location", "score", "description", "semantic_score", "keyword_score","penalty_score", "is_priority", "source", "url", "posted_date"]

    if not matches:
        print("No new matches above threshold")
        pd.DataFrame(columns=headers).to_csv(OUTPUT_CSV, mode="w", header=True, index=False)
        return


    """with open('matched_jobs.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)"""

    df = pd.DataFrame(matches)
    df = df.sort_values(by=["is_priority", "score"], ascending=[False, False])
    df = df[headers]

    #write_header = not os.path.exists(OUTPUT_CSV)
    #df.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
    df.to_csv(OUTPUT_CSV, mode="w", header=True, index=False)
    print(f"Wrote {len(matches)} new matches to {OUTPUT_CSV}")

def _prepare_applied_view(df):
    """ derive 'Days Since Applied' from 'Date' and sort ascending (fewest
    days since applied - i.e. most recently applied - first), for display
    on the Applied tab.
    """
    df = df.copy()

    if df.empty:
        if "Days Since Applied" not in df.columns:
            df["Days Since Applied"] = pd.Series(dtype="int")
        return df

    df = _days_since_applied(df)
    df = df.sort_values(by="Days Since Applied", ascending=True).reset_index(drop=True)
    return df

def append_applied_jobs_to_sheet(rows):
    """ append one row per newly-applied job to a Google Sheet:
    Title -> column A, Company -> column B, Date Applied -> column C,
    URL -> column M. Columns D-L are left blank for anything else that
    might populate them separately.

    Requires GOOGLE_SHEETS_ID to be set, and GOOGLE_SHEETS_CREDENTIALS_FILE
    to point at a service account JSON key that has been shared as an
    Editor on the target spreadsheet.

    Args:
        rows (list[dict]): applied-job rows with Title, Company, Date, URL keys

    Returns:
        bool: True if the write succeeded, False otherwise (errors are
        logged, not raised, so a Sheets failure never blocks the CSV save)
    """
    if not rows:
        return True

    if not GOOGLE_SHEETS_ID:
        print("[warning] GOOGLE_SHEETS_ID not set - skipping Google Sheets export")
        return False

    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(GOOGLE_SHEETS_CREDENTIALS_FILE, scopes=scopes)
        client = gspread.authorize(creds)
        worksheet = client.open_by_key(GOOGLE_SHEETS_ID).worksheet(GOOGLE_SHEETS_WORKSHEET)

        for row in rows:
            # columns A, B, C, D-L (blank), M -> 13 values total
            sheet_row = [
                row.get("Title", ""),
                row.get("Company", ""),
                row.get("Date Applied", ""),
                "", "", "", "", "", "", "", "", "",  # D - L left blank
                row.get("URL", ""),
            ]
            worksheet.append_row(sheet_row, value_input_option="USER_ENTERED")

        return True

    except Exception as e:
        print(f"[error] Failed to write to Google Sheet: {e}")
        return False

def load_dashboard_df():
    """ Build the dataframe backing the 'New Jobs' grid. Ensures jobs.db,
    applied_jobs.csv, and ignored_jobs.csv all exist first so this works
    no matter which button was pressed first. Falls back to the DB if
    matched_jobs.csv is empty/missing, and filters out anything already
    applied to or ignored.
    """
    ensure_applied_csv()
    ensure_ignored_csv()

    if os.path.exists(OUTPUT_CSV):
        try:
            df = pd.read_csv(OUTPUT_CSV)
            use_db = df.empty
        except pd.errors.EmptyDataError:
            use_db = True
    else:
        use_db = True

    if use_db:
        conn = init_db()  # CREATE TABLE IF NOT EXISTS so this can't 404 on a fresh run
        df = pd.read_sql_query(
            """
            SELECT job_hash, title, company, location, posted_date, score, description, url
            FROM jobs ORDER BY score DESC LIMIT 40
            """,
            conn,
        )
        conn.close()

    df = df[["job_hash","title","company","location","posted_date","score","description","url"]].copy()
    df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce").dt.strftime("%b %d")
    df.rename(columns={
        "job_hash": "id", "title": "Title", "company": "Company", "location": "Location",
        "posted_date": "Date", "score": "Score", "description": "Description", "url": "URL",
    }, inplace=True)

    df_applied = pd.read_csv(APPLIED_CSV)
    df_ignored = pd.read_csv(IGNORED_CSV)

    df = df[~df["id"].isin(df_applied["id"])].copy()
    if "id" in df_ignored.columns:
        df = df[~df["id"].isin(df_ignored["id"])].copy()

    return df

def launch_dashboard(launch=True):
    df = load_dashboard_df()
    APPLIED_DISPLAY_COLUMNS = ["id", "Title", "Company", "Location", "Date", "Score", "Description", "URL", "Date Applied", "Days Since Applied"]

    if not launch:
        return

    app = Dash()

    app.layout = dash_html.Div([
        dcc.Tabs([

            # NEW JOBS TAB
            dcc.Tab(label="New Jobs", children=[

                # toolbar
                dash_html.Div([
                    dash_html.Button("SCAN NEW JOBS", id="refresh-btn", n_clicks=0),

                    # action buttons for the currently-selected grid rows
                    dash_html.Button("Open", id="open-btn", n_clicks=0,
                                style={"marginLeft": "auto"}),
                    dash_html.Button("Apply", id="apply-btn", n_clicks=0),
                    dash_html.Button("Ignore", id="ignore-btn", n_clicks=0),

                ], style={
                    "display": "flex",
                    "gap": "10px",
                    "marginBottom": "15px",
                    "alignItems": "center",
                }),

                dash_html.Div(id="action-status", style={"marginBottom": "10px", "color": "#555"}),

                dag.AgGrid(
                    id="job-grid",
                    rowData=df.to_dict("records"),
                    columnDefs=[{"field": c, "hide": True} if c == "id" or c == "Description" else {"field": c} for c in df.columns],
                    defaultColDef={
                        "sortable": True,
                        "filter": True,
                        "resizable": True,
                    },
                    dashGridOptions={
                        "rowSelection": "multiple",
                        "suppressRowClickSelection": False,
                        "rowMultiSelectWithClick": False,
                    },
                    style={"height": "700px", "width": "100%"},
                ),

                # dummy store just to give the clientside "Open" callback somewhere to write
                dcc.Store(id="open-dummy"),
            ]),

            # APPLIED TAB
            dcc.Tab(label="Applied", children=[
                dash_html.H3("Applied Jobs"),
                dag.AgGrid(
                    id="applied-grid",
                    rowData=_load_applied(),
                    columnDefs=[
                        {"field": c, "hide": True} if c in ("id", "Description")
                        else {"field": c, "sort": "asc"} if c == "Days Since Applied"
                        else {"field": c}
                        for c in APPLIED_DISPLAY_COLUMNS
                    ],
                    defaultColDef={"sortable": True, "filter": True, "resizable": True},
                    style={"height": "700px", "width": "100%"},
                ),
            ]),

        ])
    ])

    # scan new jobs button
    @app.callback(
        Output("action-status", "children", allow_duplicate=True),
        Output("job-grid", "rowData", allow_duplicate=True),
        Input("refresh-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def scan_new_jobs(n_clicks):
        new_matches = get_new_jobs()
        df = load_dashboard_df()
        return (
            f"Scan complete: {len(new_matches)} new match(es) found.",
            df.to_dict("records"),
        )

    # open button - opens each selected row's URL in a new tab
    clientside_callback(
        """
        function(n_clicks, selectedRows) {
            if (!n_clicks || !selectedRows || selectedRows.length === 0) {
                return window.dash_clientside.no_update;
            }
            selectedRows.forEach(function(row) {
                if (row.URL) {
                    window.open(row.URL, '_blank');
                }
            });
            return window.dash_clientside.no_update;
        }
        """,
        Output("open-dummy", "data", allow_duplicate=True),
        Input("open-btn", "n_clicks"),
        State("job-grid", "selectedRows"),
        prevent_initial_call=True,
    )

    # apply button - append selected rows to applied_jobs.csv
    @app.callback(
        Output("action-status", "children", allow_duplicate=True),
        Output("applied-grid", "rowData", allow_duplicate=True),
        Output("job-grid", "rowData", allow_duplicate=True),
        Input("apply-btn", "n_clicks"),
        State("job-grid", "selectedRows"),
        State("job-grid", "rowData"),
        prevent_initial_call=True,
    )
    def apply_selected(n_clicks, selected_rows,current_rows):
        if not selected_rows:
            return "No rows selected.", no_update, no_update

        ensure_applied_csv()

        new_df = pd.DataFrame(selected_rows)
        new_df["Date Applied"] = datetime.now().strftime("%b %d")
        existing = pd.read_csv(APPLIED_CSV)

        # only rows not already recorded as applied get logged to google sheet
        already_applied_ids = set(existing["id"]) if "id" in existing.columns else set()
        rows_for_sheet = new_df[~new_df["id"].isin(already_applied_ids)].to_dict("records")

        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset="id", keep="first", inplace=True)
        combined = _days_since_applied(combined)  # recompute for all applied rows
        combined = _sort_by_date_desc(combined)
        combined.to_csv(APPLIED_CSV, index=False)

        #sheet_ok = append_applied_jobs_to_sheet(rows_for_sheet) #TODO
        #applied_view = _prepare_applied_view(combined)
        applied_view = combined

        # remove applied rows from New Jobs grid
        applied_urls = {row["URL"] for row in selected_rows}
        remaining_rows = [row for row in current_rows if row["URL"] not in applied_urls]

        return (
            f"Saved {len(new_df)} job(s) to {APPLIED_CSV}.",
            applied_view.to_dict("records"),
            remaining_rows,
        )

    # ignore button - append selected rows to applied_jobs.csv
    @app.callback(
        Output("action-status", "children", allow_duplicate=True),
        Output("applied-grid", "rowData", allow_duplicate=True),
        Output("job-grid", "rowData", allow_duplicate=True),
        Input("ignore-btn", "n_clicks"),
        State("job-grid", "selectedRows"),
        State("job-grid", "rowData"),
        prevent_initial_call=True,
    )
    def ignore_selected(n_clicks, selected_rows,current_rows):
        if not selected_rows:
            return "No rows selected.", no_update, no_update

        ensure_ignored_csv()

        new_df = pd.DataFrame(selected_rows)
        existing = pd.read_csv(IGNORED_CSV)

        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset="URL", keep="first", inplace=True)
        combined.to_csv(IGNORED_CSV, index=False)

        # remove applied rows from New Jobs grid
        applied_urls = {row["URL"] for row in selected_rows}
        remaining_rows = [row for row in current_rows if row["URL"] not in applied_urls]

        return (
            f"Saved {len(new_df)} job(s) to {IGNORED_CSV}.",
            combined.to_dict("records"),
            remaining_rows,
        )

    app.run(debug=True)

def ensure_applied_csv():
    """ create applied_jobs.csv with the correct headers if it doesn't exist yet."""
    if not os.path.exists(APPLIED_CSV):
        headers = ["id","Title", "Company", "Location", "Date", "Score", "Description", "URL", "Date Applied", "Days Since Applied"]
        pd.DataFrame(columns=headers).to_csv(APPLIED_CSV, index=False)
        print(f"Created {APPLIED_CSV}")

def ensure_ignored_csv():
    """ create ignored_jobs.csv with the correct headers if it doesn't exist yet."""
    if not os.path.exists(IGNORED_CSV):
        headers = ["id", "Title", "Company", "Location", "Date", "Score", "Description", "URL"]
        pd.DataFrame(columns=headers).to_csv(IGNORED_CSV, index=False)
        print(f"Created {IGNORED_CSV}")

def _sort_by_date_desc(df, col="Date"):
    """ sort a dataframe so the newest 'Date' (e.g. 'Jul 29') is on top.

    Note: Date has no year (format is '%b %d'), so all rows parse onto the
    same reference year. This sorts correctly within a year but can't tell
    two different years apart - fine for this app since jobs don't stick
    around that long, but worth knowing if old rows ever pile up.
    """
    # TODO
    if df.empty or col not in df.columns:
        return df
    sort_key = pd.to_datetime(df[col], format="%b %d", errors="coerce")

    return (
        df.assign(_sort_date=sort_key)
        .sort_values(by="_sort_date", ascending=False)
        .drop(columns="_sort_date")
        .reset_index(drop=True)
    )

def _days_since_applied(df):
    current_year = datetime.now().year

    df["Days Since Applied"] = (
        datetime.now()
        - pd.to_datetime(
            df["Date Applied"] + f" {current_year}",
            format="%b %d %Y"
        )
    ).dt.days
    return df

def _load_applied():
    ensure_applied_csv()
    df = pd.read_csv(APPLIED_CSV)
    #df = _days_since_applied(df)
    df = _prepare_applied_view(df)
    df = _sort_by_date_desc(df)
    return pd.read_csv(APPLIED_CSV).to_dict("records")

def get_new_jobs():
    print("Searching for matching job postings")
    all_jobs = []
    #all_jobs += get_adzuna_jobs()
    all_jobs += get_greenhouse_jobs()

    print("Filtering excluded jobs...")
    all_jobs = filter_excluded_jobs(all_jobs)


    print("Scoring against resume/cover letter...")
    resume_text = load_resume_text()
    all_jobs = score_jobs(all_jobs, resume_text)

    # save jobs to database
    conn = init_db()
    new_matches = save_new_jobs(conn, all_jobs)
    conn.close()

    write_matches_csv(new_matches)

    priority_count = sum(1 for m in new_matches if m["is_priority"])
    print(
        f"\nDone. {len(all_jobs)} total postings scanned, "
        f"{len(new_matches)} new matches saved, "
        f"{priority_count} flagged as priority."
    )

    return new_matches

def main():

    launch_dashboard()

if __name__ == "__main__":
    main()

    # TODO ADD THIS INTO MAIN, ADJUST WAY TO LAUNCH PROGRAM
