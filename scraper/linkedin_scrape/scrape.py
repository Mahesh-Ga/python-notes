import csv, os, re, sys, time
from datetime import datetime
from urllib.parse import quote

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait

TOPICS = [
    'AI engineer', 'generative AI engineer', 'GenAI engineer', 'agentic AI engineer', 'LLM engineer',
    'AI agents', 'LangChain', 'LangGraph', 'RAG', 'MCP', 'prompt engineering', 'fine tuning LLM',
]
SUFFIXES = ['interview questions', 'interview experience', 'interview asked']
QUERIES = [f'{t} {s}' for t in TOPICS for s in SUFFIXES]
MAX_SCROLLS = 25          # upper bound on scroll attempts per query
SCROLL_PAUSE_TIME = 3
OUTPUT_FILE = 'linkedin_posts.csv'

# set LINKEDIN_USER / LINKEDIN_PASS in your shell instead of hardcoding them
username = "narutouzumakkii9999@gmail.com"
password = "Mahesh@123"

# Selenium Manager down
# loads a chromedriver matching the installed Chrome
browser = webdriver.Chrome()
wait = WebDriverWait(browser, 20)


# ---------------------------------------------------------------- login
# LinkedIn's login page uses generated ids, so match by input type and pick the visible one
def visible_input(css):
    def find(d):
        for el in d.find_elements(By.CSS_SELECTOR, css):
            if el.is_displayed():
                return el
        return False
    return find


browser.get('https://www.linkedin.com/login')
try:
    elementID = wait.until(visible_input('input[type="email"], input[type="text"]'))
except Exception:
    print('Login form not found. Current page:', browser.current_url, '|', browser.title)
    print('If this is a captcha/verification page, solve it in the browser.')
    raise
elementID.send_keys(username)

elementID = wait.until(visible_input('input[type="password"]'))
elementID.send_keys(password)
elementID.send_keys(Keys.RETURN)

# wait for login to finish (also gives time for any verification step, up to 2 min)
WebDriverWait(browser, 120).until(
    lambda d: '/login' not in d.current_url and '/checkpoint' not in d.current_url)



def to_int(text):
    m = re.search(r'([\d,.]+)\s*([KkMm]?)', text or '')
    if not m:
        return ''
    num = float(m.group(1).replace(',', ''))
    return int(num * {'': 1, 'k': 1_000, 'm': 1_000_000}[m.group(2).lower()])


def button_count(post, label):
    btn = post.find('button', attrs={'aria-label': label})
    return to_int(btn.get_text(' ', strip=True)) if btn else ''



# ---------------------------------------------------------------- per-query search, scroll, parse
EXPAND_JS = """
let n = 0;
for (const b of document.querySelectorAll('button')) {
  const t = (b.innerText || '').trim().toLowerCase();
  if (/^(…|\\.\\.\\.)?\\s*(see )?more$/.test(t)) { b.click(); n++; }
  else if (t === 'show more results') { b.click(); n++; }
}
return n;
"""



def run_query(query):
    browser.get('https://www.linkedin.com/search/results/content/?keywords=' + quote(query) + '&origin=GLOBAL_SEARCH_HEADER')
    try:
        wait.until(lambda d: 'Feed post' in d.page_source)
    except Exception:
        print(f'[{query}] no posts loaded, skipping')
        return []
    last_height = browser.execute_script("return document.body.scrollHeight")
    stale = 0
    for i in range(MAX_SCROLLS):
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        browser.execute_script(EXPAND_JS)   # expand truncated posts / load more results
        new_height = browser.execute_script("return document.body.scrollHeight")
        stale = stale + 1 if new_height == last_height else 0
        if stale >= 3:
            break
        last_height = new_height
    soup = BeautifulSoup(browser.page_source, 'lxml')
    timestamp = datetime.now().isoformat(timespec='seconds')
    out = []
    for h2 in soup.find_all('h2'):
        if h2.get_text(strip=True) != 'Feed post':
            continue
        post = h2.find_parent('div')
        body = post.find(attrs={'data-testid': 'expandable-text-box'})
        text = body.get_text(chr(10), strip=True) if body else ''
        header = []
        for p in post.find_all('p'):
            if body and body in p.descendants:
                break
            t = p.get_text(' ', strip=True)
            if t:
                header.append(t)
        name = header[0] if header else ''
        rest = [t for t in header[1:] if not t.startswith('•')]
        age = rest[-1] if rest and re.fullmatch(r'\d+\s?\w{1,3}', rest[-1]) else ''
        headline = rest[0] if rest and rest[0] != age else ''
        profile = post.find('a', href=re.compile(r'linkedin\.com/(in|company|school)/'))
        post_link = post.find('a', href=re.compile(r'/(feed/update|posts)/'))
        reactions = post.find(string=re.compile(r'^\s*[\d,.]+[KkMm]?\s+reactions?\s*$'))
        hashtags = [a.get_text(strip=True) for a in post.find_all('a', attrs={'aria-label': re.compile(r'^View hashtag')})]
        out.append({
            'author': name, 'headline': headline, 'posted': age,
            'profile_url': profile['href'] if profile else '',
            'post_url': post_link['href'] if post_link else '',
            'text': text, 'hashtags': ' '.join(hashtags),
            'reactions': to_int(reactions) if reactions else button_count(post, 'Reaction button state: no reaction'),
            'comments': button_count(post, 'Comment'), 'reposts': button_count(post, 'Repost'),
            'has_video': bool(post.find('video')), 'query': query, 'scraped_at': timestamp,
        })
    return out


posts, seen = [], set()
for n, query in enumerate(QUERIES, 1):
    for p in run_query(query):
        key = p['post_url'] or p['text'][:200]
        if key and key not in seen:
            seen.add(key)
            posts.append(p)
    print(f'[{n}/{len(QUERIES)}] {query}: {len(posts)} unique posts so far', flush=True)
    if posts:   # save after every query so Ctrl+C / a crash loses nothing
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(posts[0]))
            writer.writeheader()
            writer.writerows(posts)

print(f'Saved {len(posts)} posts to {OUTPUT_FILE}' if posts else 'No posts parsed - LinkedIn markup may have changed or the page did not load.')
browser.quit()
