from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Annotated

from fastapi import FastAPI, Form,Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse




# Initialize FastAPI app    
app = FastAPI()
# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load environment variables from .env file
load_dotenv()

# loading the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
# Check if the API key is set
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")


# Initialize OpenAI client with API key from environment variables
openai = OpenAI(api_key=api_key)

chat_log = [{"role":"system","content":"You are a helpful assistant."}]
chat_history = []
image_history = []



@app.get("/", response_class=HTMLResponse)
def chat_page(request: Request):
    # Function to render the main page with the chat interface.
    '''Returns:
        HTMLResponse: The rendered HTML page for the chat interface.
    '''
    return templates.TemplateResponse("home.html", {"request":request})



@app.post("/",response_class=HTMLResponse)
def chat(request:Request, prompt: Annotated[str, Form()]):
    #Function to get a response from the OpenAI chat model.
    '''Args:
        prompt (str): The input prompt for the chat model.
    Returns:
        str: The response from the chat model.
    '''
    chat_log.append({"role": "user", "content": prompt})
  
    # Call the OpenAI chat completion API
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=chat_log,
        temperature=0.6
    )
    reply = response.choices[0].message.content
    chat_log.append({"role": "assistant", "content": reply})
    chat_history.append({"user": prompt, "assistant": reply})
    return templates.TemplateResponse("home.html", {"request": request, "chat_history": chat_history})


@app.get("/image", response_class=HTMLResponse)
def chat_page(request: Request):
    # Function to render the main page with the chat interface.
    '''Returns:
        HTMLResponse: The rendered HTML page for the chat interface.
    '''
    return templates.TemplateResponse("image.html", {"request":request})

@app.post("/image",response_class=HTMLResponse)
def image(request:Request, prompt: Annotated[str, Form()]):
    # Function to get an image from the OpenAI image generation model.
    '''Args:
        prompt (str): The input prompt for the image generation model.
    Returns:
        str: The URL of the generated image.
    '''
    response = openai.images.generate(
        prompt=prompt,
        n=1,
        size="256x256"
    )
    image_url = response.data[0].url
    image_history.append(image_url)
    return templates.TemplateResponse("image.html", {"request": request, "image_history": image_history})