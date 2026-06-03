# HUA-DIT MCP Server

MCP server for the **Department of Informatics and Telematics, Harokopio University
of Athens** (https://dit.hua.gr). Serves a cached snapshot of public department data
(courses, teaching staff, program information) over the Model Context Protocol.

## Design

Most tools read from a **local JSON snapshot** (`hua_dit_mcp/hua_data.json`) and
perform **no network access at runtime**. This keeps tool calls fast and — crucially
for benchmarking — fully reproducible. The snapshot is produced offline by the
scraper and committed to the repository.

The **single exception** is `get_latest_news`, which fetches the department news
page **live** on each call so that announcements are always current. All other
tools remain cached/reproducible.

```
scrape_hua.py  ──(run once)──>  hua_data.json  ──(read at runtime)──>  MCP tools
```

## Tools

| Tool | Description |
|---|---|
| `list_courses(semester?, course_type?)` | Undergraduate courses, filterable by semester (1-8) and type (Compulsory/Elective) — lean |
| `search_course(query)` | Search the catalog by name/keyword — lean |
| `get_course_details(course_name)` | Full course details: code, ECTS, language, prerequisites, workload, syllabus, learning outcomes, skills |
| `list_professors()` | All teaching staff with titles, emails, research fields (lean) |
| `find_professor(name)` | Find a professor by name (lean) |
| `find_professors_by_research(field)` | Find professors by research area (lean) |
| `get_professor_cv(name)` | Full CV biography + selected publications for a professor |
| `get_undergraduate_info()` | General undergraduate program info |
| `get_internship_info()` | Internship (praktiki askisi) info |
| `get_thesis_info()` | Undergraduate thesis (ptyxiaki ergasia) info |
| `get_research_areas()` | Department research orientations |
| `list_research_projects()` | Titles of the department's research projects |
| `get_postgraduate_programs()` | MSc and PhD program info |
| `get_administration()` | Department administration / governance |
| `get_services()` | Student services |
| `get_contact_info()` | Contact info (address, phone, email) |
| `get_useful_links(category?)` | Useful links / online services (e-Class, Student Information System, account, cloud, VPN, social), filterable by category |
| `find_thesis_advisor(topic, limit?)` | Ranks faculty by fit for a thesis topic (matches research field + bio + publications), with relevance score and matched keywords |
| `recommend_courses(interest, limit?)` | Ranks courses by fit for a career goal / skill / interest (matches title + skills + learning outcomes + syllabus) |
| `compute_semester_workload(semester)` | Aggregates lecture/lab/study/project hours and ECTS across a semester's courses: totals, breakdown, per-course, average, and total ECTS |
| `get_latest_news(limit)` | **LIVE** — latest news/announcements (title, date, URL), fetched in real time |

## Setup

```bash
pip install fastmcp httpx beautifulsoup4

# (Re)generate the cached snapshot from the live site:
python scrape_hua.py

# Run the server:
python -m hua_dit_mcp.server
```

## Refreshing the snapshot

The cached data reflects the department website at scrape time. To refresh:

```bash
python scrape_hua.py
```

The server picks up the new snapshot on the next tool call (no restart needed).

## Data source & ethics

Data is scraped from publicly accessible pages of dit.hua.gr that are permitted by
the site's `robots.txt`. A single request per page is issued during scraping; the
runtime server never contacts the network. Intended for academic/educational use.
