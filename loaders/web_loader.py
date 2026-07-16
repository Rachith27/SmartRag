"""
Web URL Loader for SmartRag.

This module fetches and extracts readable text from web URLs using LangChain's WebBaseLoader
(powered by BeautifulSoup), injecting standardized citation metadata.
"""

import logging
from typing import List, Union
from urllib.parse import urlparse
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader

logger = logging.getLogger(__name__)


def load_web_url(url: Union[str, List[str]]) -> List[Document]:
    """
    Fetch web page content from one or more URLs, parse HTML into clean text,
    and inject standardized citation metadata.
    
    Args:
        url (Union[str, List[str]]): Single URL string or list of URL strings.
        
    Returns:
        List[Document]: List of LangChain Documents representing the web pages.
        
    Raises:
        ValueError: If the URL is invalid or inaccessible due to network errors.
    """
    urls = [url] if isinstance(url, str) else url
    cleaned_urls = [u.strip() for u in urls if u and u.strip()]
    
    if not cleaned_urls:
        raise ValueError("No valid URL provided for loading.")
        
    try:
        logger.info(f"Fetching web content from: {cleaned_urls}")
        loader = WebBaseLoader(cleaned_urls)
        raw_docs = loader.load()
        
        if not raw_docs:
            logger.warning(f"No text content retrieved from web URLs: {cleaned_urls}")
            return []
            
        standardized_docs: List[Document] = []
        for doc in raw_docs:
            source_url = doc.metadata.get("source", cleaned_urls[0])
            # Extract domain or page title for display
            parsed = urlparse(source_url)
            title = doc.metadata.get("title") or parsed.netloc or source_url
            
            doc.metadata.update({
                "source": source_url,
                "document_name": f"Web: {title}",
                "page_number": "Web Page",
                "url": source_url
            })
            standardized_docs.append(doc)
            
        logger.info(f"Successfully loaded {len(standardized_docs)} web documents.")
        return standardized_docs
        
    except Exception as e:
        logger.error(f"Network or parsing error while loading web URL '{cleaned_urls}': {str(e)}")
        raise ValueError(f"Failed to fetch or parse web URL: {str(e)}") from e
