from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import io
import json
from inference import AutoCleanAgent

app = FastAPI(title="AutoClean AI PRO API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

agent = AutoCleanAgent()


@app.get("/")
async def root():
    return FileResponse('static/index.html')


@app.post("/api/clean")
async def clean_dataset(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
            
        report = agent.run(df)
        
        return {
            "success": True,
            "filename": file.filename,
            "initial_score": report['initial_score'],
            "final_score": report['final_score'],
            "improvement": report['improvement'],
            "steps": report['steps_taken'],
            "history": report['history'],
            "metrics": report['final_metrics'],
            "cleaned_data": json.loads(report['cleaned_dataset'].to_json(orient='records'))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def status():
    return {
        "status": "online",
        "version": "2.0.0",
        "name": "AutoClean AI PRO"
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
