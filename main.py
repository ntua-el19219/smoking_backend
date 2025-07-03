import os
os.environ["ALLOW_CHROMA_TELEMETRY"] = "FALSE" 

from fastapi import FastAPI
from apiRoutes import router

app = FastAPI()
app.include_router(router)
