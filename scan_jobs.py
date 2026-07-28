### JOB APPLICATION MATCHER
# Madelyn Redick
# July 2026

import os
from dotenv import load_dotenv
import csv
import sqlite3
import hashlib

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