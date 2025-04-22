import streamlit as st
from bs4 import BeautifulSoup
import requests
import openai
import numpy as np

# Set your OpenAI API key
openai.api_key = "YOUR_API_KEY"

# Function to scrape content from URLs
def scrape_content(urls):
    content = []
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = ' '.join(p.get_text() for p in soup.find_all('p'))
            content.append(text)
        except Exception as e:
            st.error(f"Error fetching {url}: {e}")
    return content

# Function to embed content using OpenAI
def embed_content(content):
    # Create embeddings for the input content
    response = openai.Embedding.create(
        input=content,
        model="text-embedding-ada-002"
    )
    return [embedding['embedding'] for embedding in response['data']]

# Function to find the most relevant content based on the question
def find_relevant_content(question_embedding, content_embeddings, content):
    similarities = np.dot(content_embeddings, question_embedding)
    best_index = np.argmax(similarities)
    return content[best_index]

# Streamlit app setup
st.title("Web Content Q&A Tool")
url_input = st.text_area("Enter one or more URLs (separated by commas):")
question_input = st.text_area("Ask a question based on the content:")

if st.button("Get Answer"):
    urls = [url.strip() for url in url_input.split(',')]
    content = scrape_content(urls)
    if content:
        content_embeddings = embed_content(content)
        question_embedding = embed_content([question_input])[0]
        answer = find_relevant_content(question_embedding, content_embeddings, content)
        st.success(answer)
    else:
        st.warning("No content was scraped from the provided URLs.")