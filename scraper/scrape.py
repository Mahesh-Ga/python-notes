import json, re, sys, time, difflib, urllib.robotparser
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

UA = "Mozilla/5.0 (compatible; PrepResearchBot/1.0; personal interview-prep use)"
HERE = sys.argv[1]  # scratchpad dir
seeds = json.load(open(HERE + "/seeds.json"))
robots = {}
last = {}


def allowed(url):
    p = urlparse(url)
    host = p.scheme + "://" + p.netloc
    if host not in robots:
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = requests.get(host + "/robots.txt", headers={"User-Agent": UA}, timeout=10)
            rp.parse(r.text.splitlines() if r.status_code == 200 else [])
        except Exception:
            rp.parse([])
        robots[host] = rp
    return robots[host].can_fetch(UA, url)


def fetch(url):
    host = urlparse(url).netloc
    if "raw.githubusercontent.com" not in host and not allowed(url):
        return None, "robots"
    wait = 1.5 - (time.time() - last.get(host, 0))
    if wait > 0:
        time.sleep(wait)
    last[host] = time.time()
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
    except Exception as e:
        return None, str(e)[:60]
    if r.status_code != 200:
        return None, "http %d" % r.status_code
    return r.text, "ok"


QW = re.compile(r"^(what|how|why|when|where|which|who|explain|describe|design|implement|write|tell|can|do|does|is|are|given|compare|differentiate|define|discuss|walk|build|create|list|suppose|you)\b", re.I)


def clean(s):
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(\d+[\.\)]|Q\d*[\.:\)]|Q:|[-*•#]+)\s*", "", s, flags=re.I)
    s = re.sub(r"[*_`]+", "", s).strip()
    return s


def is_q(s):
    if not (20 <= len(s) <= 300):
        return False
    return s.endswith("?") or bool(QW.match(s))


def extract_html(html):
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        t.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    out = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "li", "strong", "b", "p"]):
        txt = clean(el.get_text(" ", strip=True))
        if is_q(txt) and (txt.endswith("?") or el.name in ("h2", "h3", "h4", "h5", "li", "strong", "b")):
            out.append(txt)
    return title, out


def extract_md(text, url):
    title = url.rsplit("/", 3)[-3] + "/" + url.rsplit("/", 3)[-2] if "githubusercontent" in url else url
    out = []
    for line in text.splitlines():
        s = clean(line)
        if is_q(s) and (s.endswith("?") or re.match(r"^\s*(\d+[\.\)]|#+|[-*]|\*\*)", line)):
            out.append(s)
    return title, out


results = []
stats = []
for item in seeds:
    url, role = item["url"], item["role"]
    body, status = fetch(url)
    if body is None:
        stats.append((url, status, 0))
        print("SKIP", status, url)
        continue
    if url.endswith(".md") or "raw.githubusercontent" in url:
        title, qs = extract_md(body, url)
    else:
        title, qs = extract_html(body)
    for q in qs:
        results.append({"q": q, "role": role, "url": url, "title": title or url})
    stats.append((url, "ok", len(qs)))
    print("OK", len(qs), url)

# dedupe
seen = {}
final = []
for r in results:
    key = re.sub(r"[^a-z0-9 ]", "", r["q"].lower())
    key = re.sub(r"\s+", " ", key).strip()
    if key in seen:
        continue
    dup = False
    for k in list(seen)[-400:]:
        if abs(len(k) - len(key)) < 12 and difflib.SequenceMatcher(None, k, key).ratio() > 0.92:
            dup = True
            break
    if dup:
        continue
    seen[key] = 1
    final.append(r)

# keep LinkedIn / AmbitionBox rows added by experience_scrape.py extract
try:
    final += [r for r in json.load(open(HERE + "/questions.json", encoding="utf-8")) if r.get("experience")]
except (FileNotFoundError, ValueError):
    pass
json.dump(final, open(HERE + "/questions.json", "w", encoding="utf-8"), indent=1)
json.dump(stats, open(HERE + "/stats.json", "w"), indent=1)
print("TOTAL raw", len(results), "dedup", len(final))
