from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import uvicorn

from transformers import pipeline

app = FastAPI()

class Body(BaseModel):
    text: str

pipe = pipeline(task="text-generation", model="Qwen/Qwen2-7B-Instruct")

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs", status_code=302)

@app.post("/generate")
def generate(body: Body):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates text based on the input provided."
        },
        {
            "role": "user",
            "content": body.text
        },
    ]
    return pipe(messages)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
