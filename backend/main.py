from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def home():
    return {
        "message":"Fastapi is running"
    }
