import json, re, sys, collections, datetime
from urllib.parse import urlparse

HERE = sys.argv[1]
OUT = sys.argv[2]
qs = json.load(open(HERE + "/questions.json", encoding="utf-8"))

IMPER = re.compile(r"^(explain|describe|design|implement|write|walk|tell|compare|differentiate|define|discuss|build|create|list|how would|given|suppose|you are|your )\b", re.I)
META = re.compile(r"prepare|recruiter|hiring manager|candidates?\b|this guide|our (course|team)|sign up|subscribe|interviewers? (look|want|evaluate)|resume", re.I)
AGENT = re.compile(r"\bagent|agentic|langgraph|crewai|autogen|\bmcp\b|tool[- ]?(use|call)|function call|multi-agent|react (loop|pattern)|planner|orchestrat", re.I)
GENAI = re.compile(r"\bllm|\brag\b|retrieval[- ]augmented|prompt|fine[- ]?tun|lora|embedding|vector (db|database|store)|hallucinat|transformer|gpt|generative|diffusion|token|chunk|rlhf|context window|langchain", re.I)
FDE = re.compile(r"customer|client|stakeholder|on-?site|forward[- ]deploy|deployment engineer|solutions? architect|pilot|onboard|business problem|ambig|escalat|palantir", re.I)


def clean(q):
    for _ in range(3):
        q = re.sub(r"^\s*(Q?\d+\s*[\.\):]|Q\d*\s*[\.:\)]|[-*•#]+)\s*", "", q, flags=re.I)
    if len(q) > 110 and "?" in q:
        head = q.split("?")[0] + "?"
        if len(head) >= 20:
            q = head
    q = q.strip()
    return q


def role_for(q, seed_role):
    if AGENT.search(q):
        return "agentic"
    if GENAI.search(q):
        return "genai"
    if seed_role in ("agentic", "genai") and not re.search(r"cnn|svm|regression|decision tree|bias|variance|overfit|gradient|k-means|pca|cross-valid", q, re.I):
        return seed_role
    return "aiml"


seen = set()
rows = collections.defaultdict(list)
for x in qs:
    q = clean(x["q"])
    if len(q) < 20 or META.search(q):
        continue
    if not q.endswith("?"):
        if not re.match(r"^(explain|describe|design|implement|write a|walk me|tell me|compare|differentiate|define|discuss|how would)\b", q, re.I):
            continue
        if len(q) > 160 or ":" in q or q.count(". ") > 0:
            continue
    if len(q) > 260:
        continue
    k = re.sub(r"[^a-z0-9]", "", q.lower())
    if k in seen:
        continue
    seen.add(k)
    if x["role"] == "fde":
        continue
    rows[role_for(q, x["role"])].append((q, x["url"], x["title"]))

names = {
    "aiml": "AI Engineer",
    "fde": "Forward Deployed Engineer (FDE)",
    "agentic": "Agentic AI Engineer",
    "genai": "Generative AI Engineer",
}
dom = collections.Counter()
for r in rows:
    for q, u, t in rows[r]:
        dom[urlparse(u).netloc.replace("www.", "")] += 1

lines = [
    "# Interview Questions From Scraping",
    "",
    f"Scraped on {datetime.date.today().isoformat()} from public web pages (prep blogs, GitHub READMEs, guides), AmbitionBox and LinkedIn posts. "
    "Each question is extracted as written on the page, with the page it came from. "
    "Prep-blog and GitHub sources are not verified real interviews; AmbitionBox and LinkedIn rows are candidate-reported experiences. "
    "Pages that block bots (robots.txt / 403 / 429) and sites like Reddit were not scraped. "
    "Role grouping is a keyword guess, so some questions can sit in a neighbouring role.",
    "",
    "| Role | Questions |",
    "|---|---|",
]
for k in ("aiml", "genai", "agentic"):
    lines.append(f"| {names[k]} | {len(rows[k])} |")
lines.append(f"| **Total** | **{sum(len(v) for v in rows.values())}** |")
lines.append("")
for k in ("aiml", "genai", "agentic"):
    lines += [f"## {names[k]} ({len(rows[k])})", ""]
    for i, (q, u, t) in enumerate(rows[k], 1):
        host = urlparse(u).netloc.replace("www.", "")
        if "raw.githubusercontent.com" in host:
            p = u.split("/")
            host = f"github.com/{p[3]}/{p[4]}"
        lines.append(f"{i}. {q} — [{host}]({u})")
    lines.append("")
lines += ["## Sources index", "", "| Site | Questions |", "|---|---|"]
for d, n in dom.most_common():
    lines.append(f"| {d} | {n} |")
open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print({k: len(v) for k, v in rows.items()}, sum(len(v) for v in rows.values()))
