### JOB APPLICATION MATCHER
# Madelyn Redick
# July 2026

import os
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
    if not matches:
        print("No new matches above threshold")
        return
    headers = ["title", "company", "location", "score", "semantic_score", "keyword_score","penalty_score", "is_priority", "source", "url", "posted_date", "description"]
    with open('matched_jobs.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    df = pd.DataFrame(matches)
    df = df.sort_values(by=["is_priority", "score"], ascending=[False, False])
    df = df[headers]

    write_header = not os.path.exists(OUTPUT_CSV)
    df.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
    print(f"Wrote {len(matches)} new matches to {OUTPUT_CSV}")

def main():
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

if __name__ == "__main__":
    main()
