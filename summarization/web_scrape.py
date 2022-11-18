import requests
from bs4 import BeautifulSoup

def web_scrape(URL):
    if 'channelnewsasia' not in URL:
        return "Only supported web scraping for channelnewsasia"

    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")

    title = soup.find("meta", property="og:title")
    print(f'Title of this URL: {title}')
    # url = soup.find("meta", property="og:url")

    results = soup.find(role="main")
    news_texts = results.find_all("div", class_="text-long")
    full_para = []
    for news_text in news_texts:
        news = news_text.find_all("p")
        full_para.extend([sent.text.strip() for sent in news])
    return ' '.join(full_para)


if __name__ == '__main__':
    # URL = "https://www.channelnewsasia.com/singapore/covid19-xbb-safe-management-measures-mask-ong-ye-kung-3009186"
    URL = "https://www.channelnewsasia.com/commentary/mask-wearing-beyond-protect-covid19-health-decision-risk-perception-3007476"
    web_scrape(URL)