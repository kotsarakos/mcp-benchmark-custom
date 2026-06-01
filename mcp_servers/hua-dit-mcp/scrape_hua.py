"""
scrape_hua.py — one-shot scraper for the Department of Informatics and Telematics,
Harokopio University of Athens (https://dit.hua.gr).

Run this ONCE to produce a cached snapshot (hua_data.json) that the MCP server
reads at runtime. Re-run it whenever you want to refresh the snapshot.

    python scrape_hua.py

Design note: the MCP server itself never hits the network. All data is served
from the local JSON snapshot, which keeps tool calls fast and — more importantly
for the benchmark — fully reproducible across runs.
"""

import base64
import json
import re
import sys
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

BASE = "https://dit.hua.gr/index.php/en"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; HUA-DIT-MCP/1.0; academic research)"}
OUTPUT = Path(__file__).parent / "hua_dit_mcp" / "hua_data.json"

# Public content pages (English variants). These are allowed by robots.txt.
PAGES = {
    "courses": f"{BASE}/programmata-spoudon-2/proptyxiako/mathimata",
    "professors": f"{BASE}/parousiasi-2/didaskontes",
    "undergraduate_info": f"{BASE}/programmata-spoudon-2/proptyxiako/plirofories",
    "internship": f"{BASE}/programmata-spoudon-2/proptyxiako/praktiki-askisi",
    "thesis": f"{BASE}/programmata-spoudon-2/proptyxiako/ptyxiaki-ergasia",
    "research_areas": f"{BASE}/research-department-gr-2/research-orientations-en",
    "research_projects": f"{BASE}/research-department-gr-2/research-projects-en",
    "administration": f"{BASE}/parousiasi-2/dioikisi-main",
    "services": f"{BASE}/parousiasi-2/services-en",
    "contact": f"{BASE}/contact-department-gr-2",
    "msc": f"{BASE}/programmata-spoudon-2/msc-en",
    "phd": f"{BASE}/programmata-spoudon-2/phd-program-en",
}


def _fetch(url: str) -> str:
    resp = httpx.get(url, headers=HEADERS, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def _clean(text: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", text).strip()


def _decode_joomla_mail(tag) -> str:
    """Joomla obfuscates emails as base64 in a <joomla-hidden-mail> 'text' attr."""
    if tag is None:
        return ""
    b64 = tag.get("text", "")
    if not b64:
        return ""
    try:
        return base64.b64decode(b64).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def parse_courses(html: str) -> list:
    """Parse the undergraduate course catalog grouped by semester and type."""
    soup = BeautifulSoup(html, "html.parser")
    courses = []
    current_semester = None
    current_type = None

    container = soup.find(id="course") or soup
    for node in container.descendants:
        if not getattr(node, "get", None):
            continue
        classes = node.get("class", []) or []
        if "semester-title" in classes:
            current_semester = _clean(node.get_text())
        elif node.name == "h4":
            # "Compulsory Courses" / "Elective Courses"
            current_type = _clean(node.get_text()).replace(" Courses", "")
        elif "course-card" in classes:
            title_el = node.find(class_="course-title")
            type_el = node.find(class_="course-type")
            link_el = node.find("a", href=True)
            course_id = None
            if link_el:
                m = re.search(r"id=(\d+)", link_el["href"])
                if m:
                    course_id = m.group(1)
            if title_el:
                courses.append({
                    "title": _clean(title_el.get_text()),
                    "type": _clean(type_el.get_text()) if type_el else current_type,
                    "semester": current_semester,
                    "course_id": course_id,
                })
    return courses


def parse_professors(html: str) -> list:
    """Parse the teaching staff cards: name, title, email, research field."""
    soup = BeautifulSoup(html, "html.parser")
    profs = []
    for card in soup.find_all(class_="faculty-card"):
        h3 = card.find("h3")
        name_title = _clean(h3.get_text()) if h3 else ""
        if "," in name_title:
            name, title = name_title.split(",", 1)
            name, title = name.strip(), title.strip()
        else:
            name, title = name_title, ""

        email = _decode_joomla_mail(card.find("joomla-hidden-mail"))

        research = ""
        for p in card.find_all("p"):
            if "Research Field" in p.get_text():
                research = _clean(p.get_text()).replace("Research Field:", "").strip()

        detail_id = None
        footer_link = card.find(class_="card-footer")
        if footer_link:
            a = footer_link.find("a", href=True)
            if a:
                m = re.search(r"id=(\d+)", a["href"])
                if m:
                    detail_id = m.group(1)

        img = card.find("img", class_="faculty-image")
        photo = img.get("src") if img else None

        if name:
            profs.append({
                "name": name,
                "title": title,
                "email": email,
                "research_field": research,
                "profile_id": detail_id,
                "photo": photo,
            })
    return profs


def parse_professor_detail(html: str) -> dict:
    """Extract a professor's CV bio and selected publications from their detail page.

    Page structure: <h3>CV</h3> <bio...> <h3>Selected Publications</h3> <list...>
    """
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find(attrs={"itemprop": "articleBody"})
    if not body:
        return {"bio": "", "publications": []}

    for tag in body.find_all(["style", "script"]):
        tag.decompose()

    bio_parts = []
    section = None  # None -> header, "cv", "pubs"

    for el in body.descendants:
        name = getattr(el, "name", None)
        if name in ("h2", "h3", "h4"):
            heading = _clean(el.get_text()).lower()
            if "cv" in heading or "curriculum" in heading:
                section = "cv"
            elif "publication" in heading:
                section = "pubs"
            else:
                section = None
            continue
        if name == "p" and section == "cv":
            txt = _clean(el.get_text())
            if txt:
                bio_parts.append(txt)

    # Publications live in <div class="publications"> with items split by <br>.
    pubs = []
    pub_div = body.find(class_="publications")
    if pub_div:
        # Replace <br> with a delimiter, then split.
        for br in pub_div.find_all("br"):
            br.replace_with("\n")
        for line in pub_div.get_text().split("\n"):
            line = _clean(line)
            # Strip leading enumeration like "1. " / "12) "
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            if len(line) > 10:
                pubs.append(line)

    # Fallback: if no <p> under CV, grab text between the CV and Publications headings.
    if not bio_parts:
        full = _clean(body.get_text(separator=" "))
        m = re.search(r"\bCV\b(.*?)(Selected Publications|$)", full, re.S)
        if m:
            bio_parts = [m.group(1).strip()]

    return {"bio": " ".join(bio_parts).strip(), "publications": pubs}


def parse_course_detail(html: str) -> dict:
    """Extract a course's details: general info (code, language, prerequisites),
    workload (hours), course content (syllabus), learning outcomes, and skills."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find(attrs={"itemprop": "articleBody"})
    if not body:
        return {}

    for tag in body.find_all(["style", "script"]):
        tag.decompose()

    # General + Workload: "<b>Label:</b> value" pairs.
    fields = {}
    for b in body.find_all(["b", "strong"]):
        label = _clean(b.get_text()).rstrip(":")
        if not label:
            continue
        # value is the text right after the bold tag, up to the next tag
        value = ""
        nxt = b.next_sibling
        while nxt is not None and getattr(nxt, "name", None) is None:
            value += str(nxt)
            nxt = nxt.next_sibling
        value = _clean(re.sub(r"<[^>]+>", " ", value))
        if label:
            fields[label] = value

    # Section texts keyed by heading.
    sections = {}
    current = None
    parts = []
    for el in body.descendants:
        name = getattr(el, "name", None)
        if name in ("h2", "h3", "h4"):
            if current and parts:
                sections[current] = _clean(" ".join(parts))
            current = _clean(el.get_text())
            parts = []
        elif name in ("p", "li", "div") and current:
            txt = _clean(el.get_text())
            if txt:
                parts.append(txt)
    if current and parts:
        sections[current] = _clean(" ".join(parts))

    return {
        "code": fields.get("Code", ""),
        "language": fields.get("Language", ""),
        "delivery": fields.get("Delivery", ""),
        "prerequisites": fields.get("Prerequisites", ""),
        "workload": {
            "lectures": fields.get("Lectures", ""),
            "lab": fields.get("Lab", ""),
            "study": fields.get("Study", ""),
            "project": fields.get("Project", ""),
        },
        "course_content": sections.get("Course Content", ""),
        "learning_outcomes": sections.get("Learning Outcomes", ""),
        "skills": sections.get("Skills", ""),
    }


def parse_research_projects(html: str) -> list:
    """Extract the list of research project titles (each is an <h3> heading)."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find(attrs={"itemprop": "articleBody"}) or soup
    projects = []
    for h in body.find_all("h3"):
        title = _clean(h.get_text())
        # Skip module/sidebar titles
        if title and "sp-module" not in (h.get("class") or []):
            projects.append(title)
    return projects


def parse_article_text(html: str) -> str:
    """Extract the plain-text body of a generic article page."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find(attrs={"itemprop": "articleBody"})
    if not body:
        body = soup.find(class_="item-page") or soup
    # remove style/script
    for tag in body.find_all(["style", "script"]):
        tag.decompose()
    return _clean(body.get_text(separator=" "))


def main():
    print("Scraping dit.hua.gr (English pages)...\n")
    data = {
        "source": "https://dit.hua.gr",
        "department": "Department of Informatics and Telematics, Harokopio University of Athens",
        "language": "en",
    }

    try:
        print("  - courses ...", end=" ", flush=True)
        courses = parse_courses(_fetch(PAGES["courses"]))
        data["courses"] = courses
        print(f"{len(courses)} courses")
    except Exception as e:
        print(f"FAILED: {e}")
        data["courses"] = []

    # Enrich each course with details (code, workload, syllabus, outcomes) from
    # its detail page (one request per course — only at scrape time).
    if data.get("courses"):
        print("  - course details ...")
        ok = 0
        for c in data["courses"]:
            cid = c.get("course_id")
            if not cid:
                continue
            url = f"{BASE}?option=com_content&view=article&id={cid}"
            try:
                c["details"] = parse_course_detail(_fetch(url))
                ok += 1
            except Exception as e:
                c["details"] = {}
            time.sleep(0.4)  # be polite to the server
        print(f"      enriched {ok}/{len(data['courses'])} courses")

    try:
        print("  - professors ...", end=" ", flush=True)
        profs = parse_professors(_fetch(PAGES["professors"]))
        data["professors"] = profs
        print(f"{len(profs)} professors")
    except Exception as e:
        print(f"FAILED: {e}")
        data["professors"] = []

    # Enrich each professor with their CV bio + selected publications from the
    # detail page (one request per professor — only done at scrape time).
    if data.get("professors"):
        print("  - professor CVs ...")
        for p in data["professors"]:
            pid = p.get("profile_id")
            if not pid:
                p["bio"], p["publications"] = "", []
                continue
            url = f"{BASE}?option=com_content&view=article&id={pid}"
            try:
                detail = parse_professor_detail(_fetch(url))
                p["bio"] = detail["bio"]
                p["publications"] = detail["publications"]
                print(f"      {p['name']}: {len(detail['bio'])} chars, "
                      f"{len(detail['publications'])} pubs")
            except Exception as e:
                print(f"      {p['name']}: FAILED ({e})")
                p["bio"], p["publications"] = "", []
            time.sleep(0.5)  # be polite to the server

    # research_projects is parsed as a structured list of project titles
    try:
        print("  - research_projects ...", end=" ", flush=True)
        projects = parse_research_projects(_fetch(PAGES["research_projects"]))
        data["research_projects"] = projects
        print(f"{len(projects)} projects")
    except Exception as e:
        print(f"FAILED: {e}")
        data["research_projects"] = []

    # The remaining pages are stored as plain article text.
    text_pages = (
        "undergraduate_info", "internship", "thesis",
        "research_areas", "administration", "services", "contact", "msc", "phd",
    )
    for key in text_pages:
        try:
            print(f"  - {key} ...", end=" ", flush=True)
            text = parse_article_text(_fetch(PAGES[key]))
            data[key] = text
            print(f"{len(text)} chars")
        except Exception as e:
            print(f"FAILED: {e}")
            data[key] = ""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSnapshot written to {OUTPUT}")
    print(f"  {len(data.get('courses', []))} courses, "
          f"{len(data.get('professors', []))} professors")


if __name__ == "__main__":
    sys.exit(main())
