import os

import requests
import wget
from bs4 import BeautifulSoup
from tqdm import tqdm


def download_wiki():
    main_url = 'https://dumps.wikimedia.org/enwiki/20200201/'
    page = requests.get(main_url)
    assert page.status_code == 200

    soup = BeautifulSoup(page.content, 'html.parser')

    file_urls = []

    title_span = soup.find('span',
                           text='All pages with complete page edit history (.bz2)')
    for tag in title_span.parent():
        if tag.name == 'li':
            suffix = tag.find('a')['href']
            file_url = f'https://dumps.wikimedia.org/{suffix}'
            file_urls.append(file_url)

    data_dir = '/data4/u4921817/nos/data/wikidump'
    os.makedirs(data_dir, exist_ok=True)

    for url in tqdm(file_urls):
        filename = url.split('/')[-1]
        out_path = os.path.join(data_dir, filename)

        if not os.path.exists(out_path):
            wget.download(url, out_path)


def main():
    download_wiki()


if __name__ == '__main__':
    main()
