"""Collect candidate-reported interview questions from AmbitionBox.

Put the interview-question page URLs (one per line) in experience_urls.txt, then, from this folder:
    python experience_scrape.py fetch     # open browser, save page text to raw_experiences.jsonl
    python experience_scrape.py extract   # pull questions out of the raw text into questions.json
    python ../build_md.py .. "../../interview questions from scraping.md"

No login is needed; a Chrome window opens and reads the public pages. Scraping the site
is against its terms, so keep the page count low and use it at your own risk.
"""
import json, re, sys, time, random
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).parent
URLS_FILE = HERE / "experience_urls.txt"
RAW_FILE = HERE / "raw_experiences.jsonl"
ROOT = HERE.parent      # scraper/ - shared questions.json lives here
QUESTIONS_FILE = ROOT / "questions.json"
MAX_PAGES = 15          # pages per URL
PAUSE = (3.0, 6.0)      # random delay between actions, seconds

# Review text may also hold non-AI roles, so keep only questions that look technical.
RELEVANT = re.compile(
    r"(ai|ml|llm|llms|rag|nlp|genai|gen ai|generative|agent|agents|agentic|langchain|langgraph|"
    r"crewai|autogen|mcp|prompt|fine[- ]?tun|lora|embedding|vector|transformer|attention|token|"
    r"hallucinat|guardrail|inference|neural|deep learning|machine learning|model|rlhf|"
    r"openai|gpt|bert|chunk|retriev|semantic|multi-agent|tool call|function call|evaluation)",
    re.I,
)
QSTART = re.compile(
    r"^(what|how|why|when|where|which|who|explain|describe|design|implement|write|tell|can|do|does|"
    r"is|are|given|compare|differentiate|define|discuss|walk|build|create|list|suppose|difference)\b",
    re.I,
)
LABEL = re.compile(
    r"^(interview )?questions?( asked)?:?$|^q\s*\d*[\.:)]?$|^questions? \(\d+\)$", re.I
)
STOP_LABEL = re.compile(
    r"^(helpful|report|share|answer|interview (process|experience|rounds?)|round \d|"
    r"offer|difficulty|experience|applied|duration|was this|reply|see more|read more)\b",
    re.I,
)


def load_urls(path=None):
    path = path or URLS_FILE
    urls = []
    if not path.exists():
        return urls
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line.startswith("http"):
            urls.append(line)
    return urls


def page_url(url, n):
    """AmbitionBox paginates with ?page={n}."""
    if n == 1:
        return url
    return url + ("&" if "?" in url else "?") + f"page={n}"


EXPAND_JS = """
let n = 0;
for (const el of document.querySelectorAll('button, a, span[role=button]')) {
  const t = (el.innerText || '').trim().toLowerCase();
  if (/^(\\.\\.\\.)?\\s*(read|show|see|view) more( answers?)?$|^\\.\\.\\.more$/.test(t)) {
    try { el.click(); n++; } catch (e) {}
  }
}
return n;
"""


def wait_user(msg, secs=150):
    """Wait for the user to finish a manual step: create a file named .go here, or just wait."""
    go = HERE / ".go"
    go.unlink(missing_ok=True)
    print(f"{msg} (waiting up to {secs}s; create scraper/.go to continue sooner)", flush=True)
    end = time.time() + secs
    while time.time() < end and not go.exists():
        time.sleep(2)
    go.unlink(missing_ok=True)


def open_browser():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opts = Options()
    opts.add_argument(f"--user-data-dir={HERE / '.chrome_profile'}")
    browser = webdriver.Chrome(options=opts)
    browser.get("https://www.ambitionbox.com/")
    time.sleep(3)   # no login needed for the public interview pages
    return browser


def fetch():
    urls = list(dict.fromkeys(load_urls()))
    if not urls:
        sys.exit("No URLs. Add some to experience_urls.txt")
    browser = open_browser()

    with open(RAW_FILE, "a", encoding="utf-8") as out:
        for url in urls:
            empty = 0
            for n in range(1, MAX_PAGES + 1):
                target = page_url(url, n)
                browser.get(target)
                time.sleep(random.uniform(*PAUSE))
                title = browser.title
                if re.search(r"just a moment|access denied|verify you are|captcha", title, re.I):
                    wait_user(f"Blocked on {target} - solve it in the browser", 90)
                browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.5)
                browser.execute_script(EXPAND_JS)
                time.sleep(1.5)
                text = browser.execute_script("return document.body.innerText;") or ""
                got = len(extract_from_text(text))
                print(f"{target} -> {got} question-like lines")
                out.write(json.dumps({"url": target, "base": url, "title": title, "text": text}) + "\n")
                out.flush()
                empty = empty + 1 if got == 0 else 0
                if empty >= 2:   # ran past the last page
                    break
    browser.quit()


def clean(s):
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(\d+[\.\)]|Q\d*[\.:\)]|[-*•▪●]+)\s*", "", s, flags=re.I)
    return s.strip(" \"'“”")


def extract_from_text(text):
    """Questions from one page of review text.

    Takes (a) lines that follow an "Interview Questions" style label and (b) any line
    that is itself a question. Long run-on lines are split into sentences first.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    found, in_block = [], False
    for line in lines:
        if LABEL.match(line.strip()):
            in_block = True
            continue
        if in_block and STOP_LABEL.match(line):
            in_block = False
        parts = re.split(r"(?<=[?.:])\s+", line)
        for part in parts:
            s = clean(part)
            if not (15 <= len(s) <= 260):
                continue
            is_q = s.endswith("?") or bool(QSTART.match(s))
            if is_q or (in_block and len(s.split()) >= 4 and not s.endswith(":")):
                found.append(s)
    return found


def extract():
    if not RAW_FILE.exists():
        sys.exit("Run `fetch` first.")
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    # drop earlier experience rows so re-running does not duplicate
    questions = [q for q in questions if not q.get("experience")]
    seen, added = set(), 0
    for row in map(json.loads, RAW_FILE.read_text(encoding="utf-8").splitlines()):
        host = urlparse(row["base"]).netloc.replace("www.", "")
        for q in extract_from_text(row["text"]):
            key = re.sub(r"[^a-z0-9]", "", q.lower())
            if key in seen or not RELEVANT.search(q):
                continue
            seen.add(key)
            questions.append({"q": q, "role": "aiml", "url": row["base"],
                              "title": f"{host} candidate experiences", "experience": True})
            added += 1
    li = ROOT / "linkedin_scrape" / "linkedin_posts.csv"
    if li.exists():
        import csv
        for post in csv.DictReader(open(li, encoding="utf-8-sig")):
            for q in extract_from_text(post["text"]):
                key = re.sub(r"[^a-z0-9]", "", q.lower())
                if key in seen or not RELEVANT.search(q):
                    continue
                seen.add(key)
                url = post["post_url"] or "https://www.linkedin.com/"
                if url.startswith("/"):
                    url = "https://www.linkedin.com" + url
                questions.append({"q": q, "role": "aiml", "url": url,
                                  "title": "linkedin posts", "experience": True})
                added += 1
    QUESTIONS_FILE.write_text(json.dumps(questions, indent=1), encoding="utf-8")
    print(f"Added {added} candidate-reported questions (total {len(questions)})")


if __name__ == "__main__":
    {"fetch": fetch, "extract": extract}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
