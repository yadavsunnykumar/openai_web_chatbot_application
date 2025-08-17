from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Annotated

from fastapi import FastAPI, Form,Request,WebSocket
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

chat_log = [{"role":"system","content":"You are ChatGPT, a helpful and knowledgeable assistant.\
    You will help users with any questions or tasks they have, across a wide range of topics including but not limited to programming, science, math, history, general knowledge, entertainment, advice, and more.\
    You will provide clear, accurate, and concise explanations, and respond in a friendly and respectful manner.\
    You will adapt your responses to the user's needs and level of understanding.\
    You will not use any offensive or inappropriate language.\
    You will always aim to be helpful, informative, and engaging."}]
chat_history = []
image_history = []



@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    # Function to render the main page with the chat interface.
    '''Returns:
        HTMLResponse: The rendered HTML page for the chat interface.
    '''
    return templates.TemplateResponse("home.html", {"request":request,"chat_history": chat_history})



@app.websocket("/ws")
async def chat(websocket: WebSocket):
    await websocket.accept()
    global chat_history
    while True:
        user_input = await websocket.receive_text()
        chat_log.append({"role": "user", "content": user_input})
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=chat_log,
                temperature=0.6,
                stream=True
            )
            ai_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    ai_response += chunk.choices[0].delta.content
                    await websocket.send_text(chunk.choices[0].delta.content)

            # ✅ append instead of reset
            chat_history.append({"user": user_input, "assistant": ai_response})

        except Exception as e:
            reply = f"Error: {str(e)}"
            await websocket.send_text(reply)
            break


@app.post("/", response_class=HTMLResponse)
async def chat(request: Request, user_input: Annotated[str, Form()]):
    global chat_history
    chat_log.append({"role": "user", "content": user_input})

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=chat_log,
        temperature=0.6
    )
    reply = response.choices[0].message.content
    chat_log.append({"role": "assistant", "content": reply})

    # ✅ append instead of reset
    chat_history.append({"user": user_input, "assistant": reply})
    
    return templates.TemplateResponse("home.html", {"request": request, "chat_history": chat_history})


@app.get("/image", response_class=HTMLResponse)
async def chat_page(request: Request):
    # Function to render the main page with the chat interface.
    '''Returns:
        HTMLResponse: The rendered HTML page for the chat interface.
    '''
    return templates.TemplateResponse("image.html", {"request":request,"image_history":image_history})

@app.post("/image",response_class=HTMLResponse)
async def image(request:Request, prompt: Annotated[str, Form()]):
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