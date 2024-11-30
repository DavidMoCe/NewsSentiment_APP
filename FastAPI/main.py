# Libraries for web scraping
# Library to create the API and manage routes
from fastapi import FastAPI

# Library to make HTTP requests to websites
import requests

# Library for parsing and extracting data from HTML
from bs4 import BeautifulSoup

# Library for type hints, useful for defining lists and their contents
from typing import List

# Library to perform natural language processing (NLP) tasks, like sentiment analysis
from transformers import pipeline

# Imports the 'login' function from the huggingface_hub library to authenticate the user with Hugging Face Hub
from huggingface_hub import login

# Library to handle relative and absolute URLs
from urllib.parse import urljoin

# Initialize the FastAPI application to define and manage API routes
app = FastAPI()


# Load the pre-trained Hugging Face pipeline for sentiment analysis
# The model distilbert-base-uncased-finetuned-sst-2-english is designed for sentiment analysis in English
#sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
#sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
sentiment_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

################################################################################################################################
# Function to perform web scraping
def generic_web_content(url, tag, class1=None, class2=None, subtitle_tag=None):
    # Verify that the request was successful (status code 200)
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Error accessing the website: {response.status_code}")

    # Step 2: Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Step 3: Extract the main elements (tags with optional class)
    if class1:
        elements = soup.find_all(tag, class_=class1)
    else:
        elements = soup.find_all(tag)

    # List to store the results
    results = []
    
    # Extract title and link
    for element in elements:
        # Extract the link (the 'href' attribute value of the tag)
        relative_link = element.get("href")
        if relative_link:
            # Convert to full URL if necessary
            full_link = urljoin(url, relative_link) if relative_link.startswith('/') else relative_link

            # Extract the text from the secondary tag if specified
            if subtitle_tag:
                if class2:
                    # If class2 is a list or tuple, check if any match
                    if isinstance(class2, (list, tuple)):
                        for cls in class2:
                            subtitle = element.find(subtitle_tag, class_=cls)
                            # Break the loop if a subtitle is found
                            if subtitle:
                                break
                    # If class2 is a single class
                    else:  
                        subtitle = element.find(subtitle_tag, class_=class2)
                else:
                    subtitle = element.find(subtitle_tag)
                if subtitle:
                    text = subtitle.get_text(strip=True)
                else:
                    text = None
            else:
                text = element.get_text(strip=True)

            # Add to results if both are present
            if text and full_link:
                results.append({"title": text, "link": full_link})

    return results

# Function to removes duplicates from a list.
def remove_duplicates(lst):
    unique_list = []
    for item in lst:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list

# Function to analyze the sentiments of the news_items
def analyze_sentiments(news_items):
    # Log in using your token (only once, you can skip it if you have already logged in previously)
    login(token="<token>")

    # Analyzes the sentiments of the titles and associates them with their links
    results = []

    # Dictionary for mapping indices to labels
    #label_map = {'LABEL_0': "NEGATIVE", 'LABEL_1': "NEUTRAL", 'LABEL_2': "POSITIVE"}

    for i, r in enumerate(news_items):
        try:
            # Analyze the sentiment of the title
            sentiment = sentiment_pipeline(r['title'])

            # Get the index and the score
            #sentiment_label_index = sentiment[0]['label']
            #score = sentiment[0]['score']

            # Map the index to the corresponding label
            #if sentiment_label_index in label_map:
            #    sentiment_label = label_map[sentiment_label_index]
            #else:
            #    In case it's not in the mapping
            #    sentiment_label = "Unknown"

            # Add the result
            results.append({
                "title": r['title'],
                "link": r['link'],
                "sentiment": sentiment[0]['label'].upper(),
                "precision": round(sentiment[0]['score'] * 100, 5)
            })
        except Exception as e:
            # Handle cases where sentiment analysis or data is invalid
            print(f"Error analyzing sentiment for: {r.get('title', 'Unknown')} - {e}")
            results.append({
                "title": r.get('title', 'Unknown'),
                "link": r.get('link', 'Unknown'),
                "sentiment": "UNKNOWN",
                "confidence": 0.0
            })

    # Sort the results by precision in descending order
    if results:
        results = sorted(results, key=lambda x: x['precision'], reverse=True)
    
    return results

################################################################################################################################

# Endpoint to get the news items and their sentiments
# Scrapes headlines from BBC and performs sentiment analysis.
@app.get("/scrapping_bbc")
def scrape_bbc_headlines():
    try:
        # URL and parameters for BBC scraping
        url = "https://bbc.com"
        tag = "a"
        class1 = "sc-2e6baa30-0 gILusN"
        class2 = ["sc-8ea7699c-3 dhclWg", "sc-8ea7699c-3 hlhXXQ"]
        subtitle_tag = "h2"

        # Scraping content from BBC
        news_items = generic_web_content(url, tag, class1, class2, subtitle_tag)

        # Remove duplicates
        unique_news_items = remove_duplicates(news_items)

        # Perform sentiment analysis
        sentiment_results = analyze_sentiments(unique_news_items)

        # Return the results
        return {
            "result": sentiment_results,
            "amount": len(sentiment_results)
        }

    except Exception as e:
        return {"error": str(e)}


# CNN Endpoint
@app.get("/scrapping_cnn")
def scrape_cnn_headlines():
    try:
        # URL and parameters for CNN scraping
        url = "https://cnn.com"
        tag = "a"
        class1 = "container__link"
        class2 = "container__headline-text"
        subtitle_tag = "span"

        # Scraping content from CNN
        news_items = generic_web_content(url, tag, class1, class2, subtitle_tag)

        # Remove duplicates
        unique_news_items = remove_duplicates(news_items)

        # Perform sentiment analysis
        sentiment_results = analyze_sentiments(unique_news_items)

        # Return the results
        return {
            "result": sentiment_results,
            "amount": len(sentiment_results)
        }

    except Exception as e:
        return {"error": str(e)}

# NYTimes Endpoint
@app.get("/scrapping_nytimes")
def scrape_nytimes_headlines():
    try:
        # URL and parameters for NYTimes scraping
        url_nytimes = "https://www.nytimes.com/international/section/world?page=10"
        
        # Scraping content from NYTimes
        news_items_1 = generic_web_content(url_nytimes, tag="a", class1="css-1u3p7j1")
        news_items_2 = generic_web_content(url_nytimes, tag="a", class1="css-8hzhxf", class2="css-1j88qqx e15t083i0", subtitle_tag="h3")
        
        # Combine results
        news_items = news_items_1 + news_items_2

        # Remove duplicates
        unique_news_items = remove_duplicates(news_items)

        # Perform sentiment analysis
        sentiment_results = analyze_sentiments(unique_news_items)

        # Return the results
        return {
            "result": sentiment_results,
            "amount": len(sentiment_results)
        }

    except Exception as e:
        return {"error": str(e)}


# Endpoint for getting the credits from an API
@app.get("/")
def index():
    return {"Credits" : "Created by 'David Moreno Cerezo' and 'Jairo Andrades Bueno'"}

################################################################################################################################
