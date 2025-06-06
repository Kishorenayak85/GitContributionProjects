import requests
from bs4 import BeautifulSoup

url = 'https://careers.airbnb.com/'

print('started')

# Example function to scrape job data from a portal
def scrape_job_portal(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    jobs = []
    for job_listing in soup.find_all('div', class_='job-posting'):  # Example class
        job_title = job_listing.find('h2').text.strip()
        job_location = job_listing.find('span', class_='location').text.strip()
        salary = job_listing.find('span', class_='salary').text.strip() if job_listing.find('span', class_='salary') else 'Not Listed'
        posted_date = job_listing.find('span', class_='posted-date').text.strip()

        jobs.append({
            'title': job_title,
            'location': job_location,
            'salary': salary,
            'posted': posted_date
        })

    return jobs