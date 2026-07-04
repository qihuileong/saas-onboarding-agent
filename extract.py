from scraper import fetch_text          
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)

def extract_api_info(url):
    doc_text = fetch_text(url)          
    doc_text = doc_text[:4000]         

    prompt = f"""You are an API documentation analyst.
Read the documentation below and extract:
1. Base URL
2. Authentication method (API key, OAuth, Bearer token, or none)
3. A list of available endpoints with their HTTP method and purpose

Documentation:
{doc_text}

Give your answer as a clear, structured summary."""
    
    response = llm.invoke(prompt)       
    return response.content


if __name__ == "__main__":
    result = extract_api_info("https://jsonplaceholder.typicode.com/")
    print(result)