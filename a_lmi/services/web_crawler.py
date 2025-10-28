"""
Web Crawler Service

Scrapy-based web crawler for autonomous data gathering.
Integrates with Kafka for data ingestion pipeline.
"""

import scrapy
from scrapy.crawler import CrawlerProcess
import logging
from typing import List, Set
import json
from datetime import datetime, timezone
from kafka import KafkaProducer


class WebCrawlerSpider(scrapy.Spider):
    """
    Scrapy spider for crawling web pages.
    
    Sends crawled data to Kafka for processing into Light Tokens.
    """
    
    name = "a_lmi_crawler"
    
    def __init__(self, start_urls: List[str], config: dict, *args, **kwargs):
        """
        Initialize spider.
        
        Args:
            start_urls: List of URLs to crawl
            config: Configuration dictionary
        """
        super().__init__(*args, **kwargs)
        self.start_urls = start_urls
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Kafka producer
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=config['infrastructure']['kafka']['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # Track visited URLs
        self.visited_urls: Set[str] = set()
        self.max_pages = config['a_lmi']['perception']['web_crawler']['max_pages_per_domain']
    
    def parse(self, response):
        """
        Parse response from web page.
        
        Args:
            response: Scrapy response object
        """
        # Extract content
        title = response.css('title::text').get() or ''
        text_content = ' '.join(response.css('p::text').getall())
        
        # Send to Kafka
        data = {
            'url': response.url,
            'title': title,
            'extracted_text': text_content,
            'html_ref': None,  # Would store HTML in object storage
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'domain': response.url.split('/')[2] if len(response.url.split('/')) > 2 else ''
        }
        
        self.kafka_producer.send('web_crawler', data)
        self.logger.info(f"Crawled: {response.url}")
        
        # Follow links (respect max_pages limit)
        if len(self.visited_urls) < self.max_pages:
            for next_page in response.css('a::attr(href)').getall():
                full_url = response.urljoin(next_page)
                if full_url not in self.visited_urls:
                    self.visited_urls.add(full_url)
                    yield response.follow(next_page, self.parse)


class WebCrawlerService:
    """
    Web crawler service wrapper.
    
    Manages Scrapy crawler process and coordinates with Kafka.
    """
    
    def __init__(self, config_path: str = "infrastructure/config.yaml"):
        """
        Initialize web crawler service.
        
        Args:
            config_path: Path to configuration file
        """
        import yaml
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = logging.getLogger(__name__)
        self.process = None
    
    def crawl(self, start_urls: List[str]):
        """
        Start crawling process.
        
        Args:
            start_urls: List of URLs to start crawling from
        """
        self.logger.info(f"Starting crawl of {len(start_urls)} URLs")
        
        # Configure Scrapy
        crawler_config = {
            'USER_AGENT': 'A-LMI-WebCrawler/1.0',
            'ROBOTSTXT_OBEY': self.config['a_lmi']['perception']['web_crawler']['respect_robots_txt'],
            'DOWNLOAD_DELAY': self.config['a_lmi']['perception']['web_crawler']['delay_between_requests'],
            'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
            'LOG_LEVEL': 'INFO'
        }
        
        # Create crawler process
        self.process = CrawlerProcess(crawler_config)
        
        # Add spider
        self.process.crawl(
            WebCrawlerSpider,
            start_urls=start_urls,
            config=self.config
        )
        
        # Start crawling
        self.process.start()


def main():
    """Test web crawler."""
    # Example URLs
    start_urls = [
        'https://example.com',
        'https://wikipedia.org'
    ]
    
    service = WebCrawlerService()
    service.crawl(start_urls)


if __name__ == "__main__":
    main()

