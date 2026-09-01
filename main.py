from handlers import user, auth, clipping, admin, google_auth, clipping_advanced

from fastapi import FastAPI
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

app.include_router(auth.router)
app.include_router(user.router)
app.include_router(clipping.router)
app.include_router(admin.router)
app.include_router(google_auth.router)
# app.include_router(clipping_advanced.router)


@app.get("/")
def greet():
    return {"message": "welcome to FastAPI"} 


#uvicorn main:app --host 0.0.0.0 --port 8000 --reload