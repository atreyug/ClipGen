from handlers import user, auth, clipping, admin

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

# Create necessary directories
Path("videos").mkdir(exist_ok=True)
Path("clips").mkdir(exist_ok=True)

# Mount static files for serving clips
app.mount("/clips", StaticFiles(directory="clips"), name="clips")

# Include routers
app.include_router(auth.router)
app.include_router(user.router)
app.include_router(clipping.router)
app.include_router(admin.router)

@app.get("/")
def greet():
    return {"message": "welcome to FastAPI"}
