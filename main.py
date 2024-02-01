from fastapi import FastAPI

app = FastAPI()

@app.get("/heartbeat")
async def root():
    return {"status": "alive"}

@app.post("/generate_image")
async def generate_image():
    pass