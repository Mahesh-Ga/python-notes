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