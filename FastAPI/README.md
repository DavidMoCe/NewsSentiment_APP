# Sentiment Analysis and Web Scraping API

This project is a FastAPI-based application that performs web scraping on various news websites and conducts sentiment analysis on the extracted headlines using the Hugging Face Transformers library. [2024-11-29]

---

## Setup Guide

### Requirements
- **Python Version:** 3.12 is recommended for compatibility with the Transformers library.
- **Tools:**
  - Rust (for compiling some dependencies in Transformers)
  - Visual Studio Installer with the `Desktop development with C++` workload
  - Hugging Face token (for accessing certain models)

---

### Video Tutorial to create the API
[Crea una API con Python en menos de 5 minutos (Fast API)](https://www.youtube.com/watch?v=J0y2tjBz2Ao)

---

### Steps to Set Up the Environment

#### 1. Create a Virtual Environment
Now we will create the file necessary to build the API. Open a new terminal and create a new folder.

Once inside the folder, create a virtual environment (I'll name it `fastapi-env` to avoid conflicts with other projects):
- On Windows:
  ```bash
  python -m venv fastapi-env
  ```
- On Ubuntu/Linux:
  ```bash
  python3 -m venv fastapi-env
  ```

#### 2. Activate the Virtual Environment
- On Windows:
  ```bash
  fastapi-env\Scripts\activate
  ```
- On Ubuntu/Linux:
  ```bash
  source fastapi-env/bin/activate
  ```

#### 3. Install Dependencies
If a `requirements.txt` file exists:
```bash
pip install -r requirements.txt
```

If no `requirements.txt` is available, install the required packages manually:
```bash
pip install fastapi uvicorn requests beautifulsoup4 transformers torch
```
> [!NOTE]
> If you are using **Python 3.13**, it is recommended to use Anaconda, as its Python version is compatible with this project.

#### 4. Install Rust (Required for Transformers)
Visit the [Rust installation](https://rustup.rs/) page and follow the instructions for your platform.

---

### Optional Setup (GeoIP-Based Access)
- Install GeoIP2 for demographic-based access restriction:
  ```bash
  pip install geoip2
  ```
- Download the GeoIP database from [MaxMind's website](https://dev.maxmind.com/geoip/geolocate-an-ip/databases/).

- Code Example:
  ```python
  from fastapi import FastAPI, HTTPException, Request
  import geoip2.database
  
  app = FastAPI()
  
  # Load the pre-trained GeoIP database for location-based access
  # The GeoLite2-City.mmdb file must be downloaded from MaxMind's website
  geoip_reader = geoip2.database.Reader('GeoLite2-City.mmdb')  # Ensure this file is in your project directory
  
  # Function to check if the user is allowed based on their IP location
  def is_user_allowed(ip: str) -> bool:
      try:
          # Query the IP address using the GeoIP2 database
          response = geoip_reader.city(ip)
  
          # Check if the country is Spain (ISO code "ES")
          if response.country.iso_code != "ES":
              return False
          
          # Check if the region is Andalusia
          if response.subdivisions.most_specific.name != "Andalucía":
              return False
          
          # Check if the city is Málaga
          if response.city.name != "Málaga":
              return False
          
          return True  # If all conditions are met, allow access
      except Exception as e:
          # Handle any exception (e.g., invalid IP address or GeoIP query failure)
          return False
  
  # Endpoint to access the content and check the user's location
  @app.get("/scrapping_CNN")
  def obtener_titulares_CNN(request: Request):
      user_ip = request.client.host  # Get the IP address of the user
  
      # Check if the user is allowed based on their geographical location
      if not is_user_allowed(user_ip):
          raise HTTPException(status_code=403, detail="Access Denied: You must be from Málaga, Andalucía, Spain.")
  
      # Continue with the scraping and sentiment analysis (from your existing code)
      try:
          # Call the scraping function
          news_items = content_web_CNN()
  
          # Perform sentiment analysis on the extracted titles
          sentiment_results = analyze_sentiments_CNN(news_items)
  
          # Return the sentiment results and the titles
          return {"result": sentiment_results, "amount": len(sentiment_results)}
  
      except Exception as e:
          return {"error": str(e)}
  ```
  
---

### Hugging Face Token Setup
1. Visit [Hugging Face](https://huggingface.co/) and create an account.
2. Navigate to your [account settings](https://huggingface.co/settings/tokens) and generate a personal access token.
3. Use the token in the project by logging in (only required once):
```python
from huggingface_hub import login
login("your_huggingface_token_here")
```

---

### Running the API
1. Create a file named `main.py` containing the application code.
2. Start the FastAPI server using uvicorn:
   ```bash
   uvicorn main:app --reload
   ```
  For remote hosting (**It is recommended to change the default port** for enhanced security):
  ```bash
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
  ```
3. Deactivate the virtual enviroment when done:
   ```bash
   deactivate
   ```
   Or `CTLR + C`

If deactivation doesn't work, identify and terminate the `uvicorn` process:
```bash
tasklist | findstr uvicorn
taskkill /PID <PID>
```

---

### Generating a Requirements File
To create a `requirements.txt` file for deployment:
```bash
pip freeze > requirements.txt
```

---

### Project Features
#### API Endpoints
- GET `/scrapping_bbc`: Scrapes and analyzes headlines from BBC.
- GET `/scrapping_cnn`: Scrapes and analyzes headlines from CNN.
- GET `/scrapping_nytimes`: Scrapes and analyzes headlines from The New York Times.
- GET `/`: Displays project credits.
- GET `/docs`: Automatically Generated API Documentation by FastAPI.

#### Sentiment Analysis
The project uses Hugging Face's `yiyanghkust/finbert-tone` model for analyzing sentiment (Positive, Neutral, Negative).

We tested different models but decided to use  `yiyanghkust/finbert-tone`.

The best ones and those with the highest ratings are:
- [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
- [cardiffnlp/twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [yiyanghkust/finbert-tone](https://huggingface.co/yiyanghkust/finbert-tone?text=Watch+UK+Championship%3A+Hawkins+frame+away+from+beating+Murphy+after+Trump%27s+impressive+win+over+Zhang)

The **first model** only classifies into **positive** and **negative**, the **second model** classifies into **positive** (LABEL_2), **negative** (LABEL_0), and **neutral** (LABEL_1), but it requires an **authentication token** to use it, which can be created from [your Hugging Face account](https://huggingface.co/settings/tokens) . The **third model** is the same as the second one, it also requires the **token**, but this one labels the sentiments as **Negative**, **Positive**, and **Neutral**, not as LABEL_0, LABEL_1, and LABEL_2.

#### URL for Local Access
- [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

### Additional Notes
- Ensure that Rust and Visual Studio are properly set up to avoid dependency issues.
- Always activate your virtual environment before running or modifying the code.
- The GeoIP2 module is optional and used for restricting API access based on geographical location.

---

### Credits
This project was created by:

- David Moreno Cerezo
- Jairo Andrades Bueno
