from handlers import user, auth, clipping, admin

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Path("videos").mkdir(exist_ok=True)
Path("clips").mkdir(exist_ok=True)

app.mount("/clips", StaticFiles(directory="clips"), name="clips")

app.include_router(auth.router)
app.include_router(user.router)
app.include_router(clipping.router)
app.include_router(admin.router)

@app.get("/")
def greet():
    return {"message": "welcome to FastAPI"}
