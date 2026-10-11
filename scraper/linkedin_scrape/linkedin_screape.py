from open_linkedin_api import Linkedin
import json

# Replace with your cookie values
LI_AT = "AQEDAWKDqqcDae3EAAABm5kuXGoAAAGhUMpN4FYAdblzbwnr3akRdCgmuSNSGXdwZrFVHCtUgm-MtBg3Y06Fq7U8ZYpZ7t2T8K58VD2SYxMCQVlTRpGG7Emtr9S-Z4zzUxWGY00eWrYel6wRCgBBoDgF"
JSESSIONID = "ajax:8445934132271321639"

cookies = {
    'li_at': LI_AT,
    'JSESSIONID': JSESSIONID
}

# try:
#     # Authenticate using cookie
#     # api = Linkedin(li_at=LI_AT, jsessionid=JSESSIONID)
    
#     # api = Linkedin('narutouzumakkii9999@gmail.com', 'Mahesh@123')
#     api = Linkedin('', '', cookies=cookies)
#     # Search posts by keyword
#     print("Searching for posts...")
#     posts = api.search_('ai interview questions', count=20)
    
#     # Save and print results
#     results = []
#     for i, post in enumerate(posts, 1):
#         author = post.get('author', {}).get('name', 'Unknown')
#         text = post.get('text', '')[:100]
#         print(f"[{i}] {author}")
#         print(f"    {text}...")
#         results.append(post)
    
#     # Save to JSON file
#     with open("linkedin_posts.json", "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)
    
#     print(f"\nSuccessfully scraped {len(results)} posts, saved to linkedin_posts.json")

# except Exception as e:
#     print(f"Error: {e}")
#     print("Possible reasons: cookie expired, account flagged for rate limiting, or LinkedIn displaying a CAPTCHA challenge.")


from linkedin_api import Linkedin
import json

# Authenticate
api = Linkedin('narutouzumakkii9999@gmail.com', 'Mahesh@123')

# Search for posts
# Note: The library's general "search" method may return mixed results (people, jobs, companies)
# when using broad keywords. This is a known limitation [citation:1][citation:7].
try:
    results = api.search(
        params={'keywords': 'artificial intelligence'},
        limit=20
    )
    
    posts = []
    for item in results:
        # The structure of search results may vary
        print(f"Found: {item.get('text', 'No text')[:100]}...")
        posts.append(item)
    
    # Save to file
    with open('linkedin_posts.json', 'w', encoding='utf-8') as f:
        json.dump(posts, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(posts)} results to linkedin_posts.json")

except Exception as e:
    print(f"Error: {e}")