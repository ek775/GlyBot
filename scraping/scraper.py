import requests
from bs4 import BeautifulSoup
import pandas as pd

### Scrapers and Parsers ###
class Scraper:
    def __init__(self, url):
        self.url = url
    def scrape(self):
        response = requests.get(self.url)
        if response.status_code != 200:
            print('Failed to fetch page')
            return 'missing_chapter'
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    
class Text_extractor:
    def __init__(self, soup):
        self.soup = soup
    def extract_text(self):
        text = ''
        for paragraph in self.soup.find_all('p'):
            text += (' ' + paragraph.get_text())
        return text
    
class Book:
    def __init__(self, title):
        self.chapters = []
        self.title = title
    def add_chapter(self, chapter):
        self.chapters.append(chapter)
    def to_file(self, filename, urls):
        df = pd.DataFrame(self.chapters, columns=['Chapter'])
        labels = pd.DataFrame(urls, columns=['URL'])
        df = pd.concat([df, labels], axis=1)
        df.to_csv(filename, index=False, sep='\t')

### Main ###

urls = []
print('Reading urls from file...')
with open(file='scrape_list_for_essentials_of_glycobiology', mode='r') as file:
    for line in file:
        urls.append(line.strip())
    
print('Scraping...')
book = Book('Essentials of Glycobiology')
for i, url in enumerate(urls):
    scraper = Scraper(url)
    soup = scraper.scrape()
    if soup == 'missing_chapter':
        print('Missing chapter')
        continue
    text_extractor = Text_extractor(soup)
    text = text_extractor.extract_text()
    book.add_chapter(text)
    print(f'Chapter {i+1} done')

print('Writing to file...')
book.to_file(filename=book.title, urls=urls)
print('done')