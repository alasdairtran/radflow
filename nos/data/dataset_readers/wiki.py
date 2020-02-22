def extract_attributes(page):
    if page.endswith('all-access_spider'):
        agent = 'spider'
        source = 'all-access'
        n = len('all-access_spider') + 1

    elif page.endswith('desktop_all-agents'):
        agent = 'all-agents'
        source = 'desktop'
        n = len('desktop_all-agents') + 1

    elif page.endswith('mobile-web_all-agents'):
        agent = 'all-agents'
        source = 'mobile-web'
        n = len('mobile-web_all-agents') + 1

    elif page.endswith('all-access_all-agents'):
        agent = 'all-agents'
        source = 'all-access'
        n = len('all-access_all-agents') + 1

    page = page[:-n]

    if page.endswith('_commons.wikimedia.org'):
        m = len('_commons.wikimedia.org')
        endpoint = 'https://commons.wikimedia.org/w/api.php'
        title = page[:-m]
        lang = 'en'
        domain = 'commons.wikimedia.org'

    elif page.endswith('_www.mediawiki.org'):
        m = len('_www.mediawiki.org')
        endpoint = 'https://www.mediawiki.org/w/api.php'
        title = page[:-m]
        lang = 'en'
        domain = 'www.mediawiki.org'

    elif page.endswith('wikipedia.org'):
        m = len('wikipedia.org') + 1
        page = page[:-m]
        # Possible languages: {'de', 'en', 'es', 'fr', 'ja', 'ru', 'zh'}
        lang = page.split('_')[-1]
        title = page[:-3]
        endpoint = f'https://{lang}.wikipedia.org/w/api.php'
        domain = f'{lang}.wikipedia'

    return {
        'agent': agent,
        'source': source,
        'endpoint': endpoint,
        'title': title,
        'lang': lang,
        'domain': domain,
    }
