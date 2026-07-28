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
from datetime import datetime
import pandas as pd

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

GREENHOUSE_COMPANIES = [
    c.strip() for c in os.getenv("GREENHOUSE_COMPANIES", "").split(",") if c.strip()
]

RESUME_FILE = os.getenv("RESUME_FILE", "resume_and_cover_letter.txt")
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.15"))
PRIORITY_KEYWORDS = [k.strip().lower() for k in os.getenv("PRIORITY_KEYWORDS", "").split(",") if k.strip()]

headers = ["title","company","location","score","is_priority","source","url","posted_date"]
with open('matched_jobs.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(headers)

OUTPUT_CSV = os.getenv("OUTPUT_CSV", "matched_jobs.csv")
DB_FILE = os.getenv("DB_FILE", "jobs.db")


# set up database
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
            score REAL,
            is_priority INTEGER,
            status TEXT DEFAULT 'new',
            keyword_score REAL,
            semantic_score REAL,
            penalty_score REAL,
            final_score REAL
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
                    "description": r.get("description", ""),
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
            jobs.append(
                {
                    "title": r.get("title", "").strip(),
                    "company": company,
                    "location": (r.get("location") or {}).get("name", ""),
                    "description": r.get("content", ""),
                    "url": r.get("absolute_url", ""),
                    "source": "greenhouse",
                    "posted_date": (r.get("updated_at") or "")[:10],
                }
            )

    print(f"Greenhouse: fetched {len(jobs)} postings across "f"{len(GREENHOUSE_COMPANIES)} companies")

    return jobs


def load_resume_text():
    with open(RESUME_FILE, "r", encoding="utf-8") as f:
        return f.read()

def score_jobs(jobs, resume_text):
    """ score each job posting based on its similarity to resume using TF-IDF vectorization and cosine similarity. each job is assigned a similarity score, jobs containing any configured priority keywords are flagged

    Args:
        jobs (list[dict]): list of job postings to score
        resume_text (str): text extracted from resume

    Returns:
        jobs (list[dict]): original list of jobs with two additional fields:
            - score (float): similarity score between resume and job
            - is_priority (bool): True if a priority keyword is found in job title or description
    """

    if not jobs:
        return jobs

    # build text corpus containing resume and all job descriptions
    descriptions = [j["description"] or "" for j in jobs]
    corpus = [resume_text] + descriptions

    # convert corpus into TF-IDF vectors, remove stop words
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vectorizer.fit_transform(corpus)

    # separate resume vector from job vectors
    resume_vec = tfidf[0:1]
    job_vecs = tfidf[1:]

    # compute cosine similarity between resume and each job description
    sims = cosine_similarity(resume_vec, job_vecs).flatten()

    # store each similarity score, check for priority keywords
    for job, sim in zip(jobs, sims):
        job["score"] = round(float(sim), 4)
        text_lower = (job["title"] + " " + job["description"]).lower()
        job["is_priority"] = any(
            kw in text_lower for kw in PRIORITY_KEYWORDS
        )

    return jobs

def save_new_jobs(conn, jobs):
    new_matches = []
    cur = conn.cursor()
    for j in jobs:
        h = job_hash(j["title"], j["company"], j["posted_date"])
        cur.execute("SELECT 1 FROM jobs WHERE job_hash = ?", (h,))
        if cur.fetchone():
            continue  # already seen this posting

        cur.execute(
            """
            INSERT INTO jobs
                (job_hash, title, company, location, description, url,
                 source, posted_date, scraped_date, score, is_priority, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new')
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
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                j["score"],
                int(j["is_priority"]),
            ),
        )
        if j["score"] >= MATCH_THRESHOLD or j["is_priority"]:
            new_matches.append(j)

    conn.commit()
    return new_matches


def write_matches_csv(matches):
    if not matches:
        print("No new matches above threshold")
        return

    df = pd.DataFrame(matches)
    df = df.sort_values(by=["is_priority", "score"], ascending=[False, False])
    cols = ["title", "company", "location", "score", "is_priority", "source", "url", "posted_date"]
    df = df[cols]

    write_header = not os.path.exists(OUTPUT_CSV)
    df.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
    print(f"Wrote {len(matches)} new matches to {OUTPUT_CSV}")

def main():
    print("Searching for matching job postings")
    all_jobs = []
    all_jobs += get_adzuna_jobs()
    all_jobs += get_greenhouse_jobs()

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

    # TODO ADD GREENHOUSE JOBS and improve matching and ranking
