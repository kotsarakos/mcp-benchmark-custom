"""
HUA-DIT MCP Server — Department of Informatics and Telematics,
Harokopio University of Athens.

Most tools serve a cached snapshot of public department data (hua_data.json),
produced offline by scrape_hua.py — fast and fully reproducible. The single
exception is get_latest_news, which fetches the department news page LIVE so
that announcements are always current.

Run:
    python -m hua_dit_mcp.server
"""

import json
import re
from pathlib import Path
from typing import Annotated

import httpx
from bs4 import BeautifulSoup
from fastmcp import FastMCP
from pydantic import Field

app: FastMCP = FastMCP("HUA Informatics & Telematics MCP Server")

_DATA_PATH = Path(__file__).parent / "hua_data.json"

# Live endpoint for the news tool (the only tool that hits the network).
_BASE = "https://dit.hua.gr/index.php/en"
_NEWS_URL = f"{_BASE}/enimerosi-2/department-news-en"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; HUA-DIT-MCP/1.0; academic research)"}


# Curated useful links / online services of the department.
# These are stable institutional URLs, verified from the live dit.hua.gr site,
# kept inline (cached-style) so the tool needs no network access at runtime.
_USEFUL_LINKS = [
    {"name": "Department website", "url": "https://dit.hua.gr/index.php/en",
     "category": "general", "description": "Official website of the Department of Informatics and Telematics (English)"},
    {"name": "e-Class", "url": "https://eclass.hua.gr",
     "category": "academic", "description": "Course management / e-learning platform"},
    {"name": "Student Information System (Students)", "url": "https://students.sis.hua.gr",
     "category": "academic", "description": "Student portal: enrollment, grades, certificates"},
    {"name": "Student Information System (Instructors)", "url": "https://teachers.sis.hua.gr",
     "category": "academic", "description": "Instructor portal of the Student Information System"},
    {"name": "myaccount", "url": "https://accounts.ditapps.hua.gr",
     "category": "services", "description": "Department account management (credentials, services)"},
    {"name": "Nextcloud", "url": "https://mycloud.ditapps.hua.gr",
     "category": "services", "description": "Department cloud file storage (Nextcloud)"},
    {"name": "mydep platform", "url": "https://mydep.ditapps.hua.gr",
     "category": "services", "description": "Department platform for students and staff"},
    {"name": "Wireguard VPN", "url": "https://www.hua.gr/portal/vpn/",
     "category": "services", "description": "University VPN access (WireGuard)"},
    {"name": "Facebook", "url": "https://www.facebook.com/ditharokopio",
     "category": "social", "description": "Department Facebook page"},
    {"name": "Instagram", "url": "https://instagram.com/dithua",
     "category": "social", "description": "Department Instagram account"},
    {"name": "LinkedIn", "url": "https://www.linkedin.com/company/77699385",
     "category": "social", "description": "Department LinkedIn page"},
    {"name": "YouTube", "url": "https://www.youtube.com/channel/UCEHkYirpXF1nSLxDCrfDZ4A",
     "category": "social", "description": "Department YouTube channel"},
]


def _load() -> dict:
    """Load the cached snapshot. Read on each call so a re-scrape is picked up
    without restarting the server, while still avoiding any network I/O."""
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


# --- Lightweight keyword scoring (used by the matching/recommendation tools) ---

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "will",
    "have", "has", "into", "their", "they", "you", "your", "our", "about", "via",
    "use", "used", "using", "based", "study", "course", "courses", "research",
    "field", "fields", "work", "topic", "topics", "area", "areas", "interest",
    "interested", "want", "would", "like", "professor", "advisor", "thesis",
}


def _tokens(text: str) -> list[str]:
    """Split a query into meaningful lowercase keywords (>=3 chars, no stopwords)."""
    raw = re.findall(r"[a-zA-ZͰ-Ͽ]+", (text or "").lower())
    return [w for w in raw if len(w) >= 3 and w not in _STOPWORDS]


def _score(query_tokens: list[str], *fields: str) -> tuple[int, int, list[str]]:
    """Score an entity against query tokens over one or more text fields.

    Returns (distinct_matches, total_occurrences, matched_keywords). The number of
    distinct matched keywords is the primary relevance signal; total occurrences is
    used only as a tie-breaker."""
    haystack = " ".join(f or "" for f in fields).lower()
    matched, occurrences = [], 0
    for tok in set(query_tokens):
        n = haystack.count(tok)
        if n:
            matched.append(tok)
            occurrences += n
    return len(matched), occurrences, sorted(matched)


def _hours(value: str) -> float:
    """Extract the numeric hours from a workload string like '39.0 hours'."""
    m = re.search(r"[\d.]+", value or "")
    return float(m.group()) if m else 0.0


def _lean_course(c: dict) -> dict:
    """Course record without the heavy details. Use get_course_details for the
    full syllabus, workload, learning outcomes, etc."""
    return {
        "title": c.get("title"),
        "type": c.get("type"),
        "semester": c.get("semester"),
        "code": (c.get("details") or {}).get("code", ""),
    }


@app.tool()
def list_courses(
    semester: int | None = Field(
        default=None,
        description="Filter by semester number (1-8). Omit for all semesters.",
    ),
    course_type: str | None = Field(
        default=None,
        description="Filter by type: 'Compulsory' or 'Elective'. Omit for all.",
    ),
) -> dict:
    """List undergraduate courses of the Department of Informatics and Telematics,
    optionally filtered by semester and/or course type (Compulsory/Elective).
    For a course's full syllabus and details, use get_course_details."""
    data = _load()
    courses = data.get("courses", [])

    if semester is not None:
        target = f"Semester {semester}"
        courses = [c for c in courses if c.get("semester") == target]

    if course_type is not None:
        ct = course_type.strip().lower()
        courses = [c for c in courses if (c.get("type") or "").lower() == ct]

    return {
        "department": data.get("department"),
        "count": len(courses),
        "filters": {"semester": semester, "course_type": course_type},
        "courses": [_lean_course(c) for c in courses],
    }


@app.tool()
def search_course(
    query: Annotated[str, Field(description="Course name or keyword to search for")],
) -> dict:
    """Search the course catalog by name or keyword (case-insensitive substring match).
    For a course's full syllabus and details, use get_course_details."""
    data = _load()
    q = query.strip().lower()
    matches = [_lean_course(c) for c in data.get("courses", []) if q in c.get("title", "").lower()]
    return {"query": query, "count": len(matches), "courses": matches}


@app.tool()
def get_course_details(
    course_name: Annotated[str, Field(description="Full or partial course name, e.g. 'Discrete Mathematics'")],
) -> dict:
    """Get the full details of a course by name (case-insensitive substring match):
    code, language, delivery mode, prerequisites, workload (hours), course content
    (syllabus), learning outcomes, and skills."""
    data = _load()
    q = course_name.strip().lower()
    results = []
    for c in data.get("courses", []):
        if q in c.get("title", "").lower():
            det = c.get("details") or {}
            results.append({
                "title": c.get("title"),
                "type": c.get("type"),
                "semester": c.get("semester"),
                **det,
            })
    return {"query": course_name, "count": len(results), "results": results}


def _lean(p: dict) -> dict:
    """Professor record without the heavy bio/publications fields.
    Use get_professor_cv for the full CV."""
    return {
        "name": p.get("name"),
        "title": p.get("title"),
        "email": p.get("email"),
        "research_field": p.get("research_field"),
    }


@app.tool()
def list_professors() -> dict:
    """List all teaching staff of the department, with their titles, emails,
    and research fields. For a professor's full CV and publications, use
    get_professor_cv."""
    data = _load()
    profs = [_lean(p) for p in data.get("professors", [])]
    return {
        "department": data.get("department"),
        "count": len(profs),
        "professors": profs,
    }


@app.tool()
def find_professor(
    name: Annotated[str, Field(description="Full or partial name of the professor")],
) -> dict:
    """Find a professor by name (case-insensitive substring match) and return their
    title, email, and research field. For the full CV, use get_professor_cv."""
    data = _load()
    q = name.strip().lower()
    matches = [_lean(p) for p in data.get("professors", []) if q in p.get("name", "").lower()]
    return {"query": name, "count": len(matches), "professors": matches}


@app.tool()
def find_professors_by_research(
    field: Annotated[str, Field(description="Research area keyword, e.g. 'machine learning', 'security', 'databases'")],
) -> dict:
    """Find professors whose research field matches a keyword (case-insensitive)."""
    data = _load()
    q = field.strip().lower()
    matches = [
        _lean(p) for p in data.get("professors", [])
        if q in (p.get("research_field") or "").lower()
    ]
    return {"query": field, "count": len(matches), "professors": matches}


@app.tool()
def get_professor_cv(
    name: Annotated[str, Field(description="Full or partial name of the professor")],
) -> dict:
    """Get a professor's full CV biography and selected publications by name
    (case-insensitive substring match)."""
    data = _load()
    q = name.strip().lower()
    matches = [p for p in data.get("professors", []) if q in p.get("name", "").lower()]
    cvs = [
        {
            "name": p.get("name"),
            "title": p.get("title"),
            "research_field": p.get("research_field"),
            "bio": p.get("bio", ""),
            "publications": p.get("publications", []),
        }
        for p in matches
    ]
    return {"query": name, "count": len(cvs), "results": cvs}


@app.tool()
def get_undergraduate_info() -> dict:
    """Get general information about the undergraduate program of the department."""
    data = _load()
    return {"topic": "Undergraduate Program", "content": data.get("undergraduate_info", "")}


@app.tool()
def get_internship_info() -> dict:
    """Get information about the internship (praktiki askisi) of the department."""
    data = _load()
    return {"topic": "Internship", "content": data.get("internship", "")}


@app.tool()
def get_thesis_info() -> dict:
    """Get information about the undergraduate thesis (ptyxiaki ergasia) requirements
    and procedure of the department."""
    data = _load()
    return {"topic": "Undergraduate Thesis", "content": data.get("thesis", "")}


@app.tool()
def get_research_areas() -> dict:
    """Get the research orientations / areas of the department."""
    data = _load()
    return {"topic": "Research Areas", "content": data.get("research_areas", "")}


@app.tool()
def list_research_projects() -> dict:
    """List the titles of the department's research projects."""
    data = _load()
    projects = data.get("research_projects", [])
    return {"count": len(projects), "projects": projects}


@app.tool()
def get_postgraduate_programs() -> dict:
    """Get information about the department's postgraduate programs:
    the MSc program and the PhD program."""
    data = _load()
    return {
        "msc": data.get("msc", ""),
        "phd": data.get("phd", ""),
    }


@app.tool()
def get_administration() -> dict:
    """Get information about the department's administration and governance
    (chair, committees, secretariat)."""
    data = _load()
    return {"topic": "Administration", "content": data.get("administration", "")}


@app.tool()
def get_services() -> dict:
    """Get information about the student services offered by the department."""
    data = _load()
    return {"topic": "Services", "content": data.get("services", "")}


@app.tool()
def get_contact_info() -> dict:
    """Get the department's contact information (address, phone, email, secretariat)."""
    data = _load()
    return {"topic": "Contact", "content": data.get("contact", "")}


@app.tool()
def get_useful_links(
    category: str | None = Field(
        default=None,
        description="Filter by category: 'general', 'academic', 'services', or 'social'. Omit for all.",
    ),
) -> dict:
    """Get useful links and online services of the department: the e-Class e-learning
    platform, the Student Information System (students/instructors), account management,
    cloud storage, VPN, and the department's social media. Optionally filter by category
    ('general', 'academic', 'services', 'social')."""
    links = _USEFUL_LINKS
    if category is not None:
        c = category.strip().lower()
        links = [l for l in links if l["category"] == c]
    return {"count": len(links), "filter": category, "links": links}


@app.tool()
def find_thesis_advisor(
    topic: Annotated[str, Field(description="The thesis topic or research interest, e.g. 'federated learning for healthcare' or 'network security'")],
    limit: int = Field(default=5, description="Maximum number of candidate advisors to return."),
) -> dict:
    """Recommend the most suitable thesis advisors for a given topic. Cross-references
    each professor's research field, biography, and publications against the topic and
    returns a ranked list with a relevance score and the keywords that matched. Use this
    to find which faculty member's expertise best fits a proposed undergraduate or
    postgraduate thesis subject."""
    data = _load()
    qt = _tokens(topic)
    if not qt:
        return {"topic": topic, "count": 0, "advisors": [], "note": "No meaningful keywords in topic."}

    ranked = []
    for p in data.get("professors", []):
        pubs = " ".join(p.get("publications", []) or [])
        n, occ, matched = _score(qt, p.get("research_field"), p.get("bio"), pubs)
        if n:
            ranked.append({
                "name": p.get("name"),
                "title": p.get("title"),
                "email": p.get("email"),
                "research_field": p.get("research_field"),
                "relevance_score": n,
                "matched_keywords": matched,
                "_occ": occ,
            })

    ranked.sort(key=lambda r: (r["relevance_score"], r["_occ"]), reverse=True)
    for r in ranked:
        r.pop("_occ", None)
    return {"topic": topic, "count": len(ranked[:limit]), "advisors": ranked[:limit]}


@app.tool()
def recommend_courses(
    interest: Annotated[str, Field(description="A career goal, skill, or area of interest, e.g. 'machine learning engineer', 'cybersecurity', 'web development'")],
    limit: int = Field(default=8, description="Maximum number of courses to recommend."),
) -> dict:
    """Recommend undergraduate courses that match a stated interest, career goal, or
    desired skill set. Scores each course against the interest using its title, skills,
    learning outcomes, and syllabus (course content), and returns a ranked list with a
    relevance score and the matched keywords."""
    data = _load()
    qt = _tokens(interest)
    if not qt:
        return {"interest": interest, "count": 0, "courses": [], "note": "No meaningful keywords in interest."}

    ranked = []
    for c in data.get("courses", []):
        det = c.get("details") or {}
        n, occ, matched = _score(
            qt,
            c.get("title"),
            det.get("skills"),
            det.get("learning_outcomes"),
            det.get("course_content"),
        )
        if n:
            ranked.append({
                "title": c.get("title"),
                "type": c.get("type"),
                "semester": c.get("semester"),
                "code": det.get("code", ""),
                "relevance_score": n,
                "matched_keywords": matched,
                "_occ": occ,
            })

    ranked.sort(key=lambda r: (r["relevance_score"], r["_occ"]), reverse=True)
    for r in ranked:
        r.pop("_occ", None)
    return {"interest": interest, "count": len(ranked[:limit]), "courses": ranked[:limit]}


@app.tool()
def compute_semester_workload(
    semester: Annotated[int, Field(description="Semester number (1-8) to compute the total workload for.")],
) -> dict:
    """Compute the aggregate teaching/study workload of a semester. Sums the lecture,
    lab, study, and project hours across all courses of the given semester and reports
    totals, the per-component breakdown, the number of courses, and the average workload
    per course."""
    data = _load()
    target = f"Semester {semester}"
    courses = [c for c in data.get("courses", []) if c.get("semester") == target]

    components = ("lectures", "lab", "study", "project")
    totals = {k: 0.0 for k in components}
    per_course = []
    for c in courses:
        wl = (c.get("details") or {}).get("workload") or {}
        course_hours = {k: _hours(wl.get(k, "")) for k in components}
        course_total = sum(course_hours.values())
        for k in components:
            totals[k] += course_hours[k]
        per_course.append({
            "title": c.get("title"),
            "code": (c.get("details") or {}).get("code", ""),
            "hours": course_hours,
            "total_hours": round(course_total, 1),
        })

    grand_total = round(sum(totals.values()), 1)
    n = len(courses)
    return {
        "semester": semester,
        "course_count": n,
        "total_hours": grand_total,
        "breakdown_hours": {k: round(v, 1) for k, v in totals.items()},
        "average_hours_per_course": round(grand_total / n, 1) if n else 0.0,
        "courses": per_course,
    }


@app.tool()
def get_latest_news(
    limit: int = Field(
        default=10,
        description="Maximum number of news items to return (most recent first).",
    ),
) -> dict:
    """Get the department's latest news and announcements, fetched LIVE from the
    website in real time. Each item has a title, publication date, and URL.

    Note: unlike the other tools (which serve a cached snapshot), this tool always
    queries the live website, so results reflect the current state of the news page."""
    try:
        resp = httpx.get(_NEWS_URL, headers=_HEADERS, timeout=20, follow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        return {"error": f"Could not fetch live news: {e}", "news": []}

    soup = BeautifulSoup(resp.text, "html.parser")
    items = []
    for art in soup.find_all("article", class_="announcement-item"):
        title_el = art.find(class_="announcement-title")
        meta_el = art.find(class_="announcement-meta")
        intro_el = art.find(class_="announcement-intro")

        link = title_el.find("a", href=True) if title_el else None
        title = " ".join(title_el.get_text().split()) if title_el else ""
        date = ""
        if meta_el:
            # strip the leading calendar emoji / icon
            date = "".join(ch for ch in meta_el.get_text() if ch.isdigit() or ch == "/").strip()
        url = link["href"] if link else ""
        if url and url.startswith("/"):
            url = "https://dit.hua.gr" + url

        if title:
            items.append({
                "title": title,
                "date": date,
                "url": url,
                "intro": " ".join(intro_el.get_text().split()) if intro_el else "",
            })

    return {"source": "live", "count": len(items[:limit]), "news": items[:limit]}


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
