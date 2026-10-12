# Interview question scraper

Builds `../interview questions from scraping.md` from public prep pages.

## Files
- `seeds.json` – list of `{url, role}` pages to scrape (role: aiml, genai, agentic, fde). Add more here.
- `scrape.py` – fetches each seed (respects robots.txt, 1.5 s delay per host), extracts question-like lines, dedupes, writes `questions.json` and `stats.json`.
- `build_md.py` – cleans, re-classifies roles by keyword, writes the markdown file with source links.

## Run
```
pip install -r requirements.txt
python scrape.py .
python build_md.py . "../interview questions from scraping.md"
```

## Not covered
Reddit, Blind, Glassdoor, LinkedIn (login/robots/ToS), and pages that return 403/429 (Medium, LeetCode, Dataford).


For Linkedin: https://chat.deepseek.com/share/g75ugf0sk0beg2w5yr
## AmbitionBox experiences
Needs your own login (done by hand in the browser; no password stored). Against the site's terms, so keep volumes low.
1. `python experience_scrape.py discover` crawls the seeds in `discover_seeds.txt` and writes every interview page it finds to `discovered_urls.txt` (add your own URLs to `experience_urls.txt` too).
2. `python experience_scrape.py fetch` (visits all discovered pages) (log in when prompted; solve captchas if shown)
3. `python experience_scrape.py extract` then re-run `build_md.py`.
Selectors are text-based and untested against live pages; if a site's layout changes, check `raw_experiences.jsonl` and adjust `extract_from_text`.
