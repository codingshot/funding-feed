import feedparser
import requests
from bs4 import BeautifulSoup
import json
import PyRSS2Gen
import datetime
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext, Document
from langchain.chat_models import ChatOpenAI
import os
import re
from collections import Counter

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Predefined categories
predefined_categories = {
    'Web3': ['blockchain', 'cryptocurrency', 'decentralized', 'NFT', 'DApp'],
    'Biotech': ['biotechnology', 'pharmaceutical', 'clinical', 'gene therapy'],
    'Artificial Intelligence': ['AI', 'machine learning', 'deep learning', 'natural language processing'],
    'Fintech': ['finance', 'cryptocurrency', 'banking', 'investment'],
    'Healthcare': ['health', 'medical', 'telemedicine', 'healthcare technology'],
    'E-commerce': ['retail', 'online shop', 'marketplace', 'e-commerce'],
    'EdTech': ['education', 'learning', 'online courses', 'teaching technology'],
    'Gaming': ['video games', 'gaming', 'mobile games', 'eSports'],
    'SaaS': ['software', 'cloud', 'platform', 'SaaS'],
    'Clean Tech': ['renewable energy', 'sustainability', 'environmental', 'clean technology']
}

def scrape_google_news(query="startup funding"):
    rss_url = f'https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en'
    feed = feedparser.parse(rss_url)
    
    articles = []
    for entry in feed.entries[:20]:  # Limit to 20 articles
        article = {
            'title': entry.title,
            'link': entry.link,
            'published': entry.published,
            'source': entry.source.title if 'source' in entry else None,
            'description': entry.summary
        }
        articles.append(article)
    
    return articles

def extract_funding_details(article):
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    
    prompt = f"""
    Extract the following information from the funding announcement:
    - Project name
    - Amount raised
    - Lead investors
    - Other backers
    - Funding stage (pre-seed, seed, series A, etc.)
    - Use of funds
    - Company's Twitter account

    Article: {article['title']} {article['description']}

    Return the information in JSON format.
    """
    
    index = GPTVectorStoreIndex.from_documents([Document(text=prompt)], service_context=service_context)
    response = index.query(prompt)
    
    details = json.loads(response.response)
    details['category'] = autotag_category(details)
    details['link'] = article['link']
    details['source'] = article['source']
    details['published'] = article['published']
    
    return details

def autotag_category(announcement):
    text = f"{announcement['project']} {announcement.get('use_of_funds', '')}"
    text = text.lower()
    
    category_scores = Counter()
    
    for category, keywords in predefined_categories.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text):
                category_scores[category] += 1
    
    if category_scores:
        top_category = max(category_scores, key=category_scores.get)
        return top_category
    else:
        return "Uncategorized"

def remove_duplicates(announcements):
    unique_announcements = []
    seen = set()
    
    for announcement in announcements:
        key = (announcement['project'], announcement['amount_raised'])
        if key not in seen:
            seen.add(key)
            unique_announcements.append(announcement)
    
    return unique_announcements

def create_rss_feed(announcements):
    items = []
    for announcement in announcements:
        item = PyRSS2Gen.RSSItem(
            title = f"{announcement['project']} raises {announcement['amount_raised']}",
            link = announcement['link'],
            description = json.dumps(announcement, indent=2),
            guid = PyRSS2Gen.Guid(announcement['link']),
            pubDate = datetime.datetime.strptime(announcement['published'], "%a, %d %b %Y %H:%M:%S %Z")
        )
        items.append(item)
    
    rss = PyRSS2Gen.RSS2(
        title = "Funding Announcements",
        link = "https://example.com/funding-announcements",
        description = "Latest funding announcements in the startup world",
        lastBuildDate = datetime.datetime.now(),
        items = items
    )
    
    rss.write_xml(open("funding_announcements.xml", "w"))

def main():
    articles = scrape_google_news()
    processed_announcements = []
    
    for article in articles:
        try:
            details = extract_funding_details(article)
            processed_announcements.append(details)
        except Exception as e:
            print(f"Error processing article: {article['title']}. Error: {str(e)}")
    
    unique_announcements = remove_duplicates(processed_announcements)
    create_rss_feed(unique_announcements)

if __name__ == "__main__":
    main()