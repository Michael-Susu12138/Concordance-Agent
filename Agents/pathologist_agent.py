from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PathologyRequest(BaseModel):
    Case_id: int
    Biopsy_benign: int
    Biopsy_malignant: int

class PathologyResponse(BaseModel):
    Case_id: int
    pathology: str
    note: str

@app.post("/analyze_pathology", response_model=PathologyResponse)
async def analyze_pathology(req: PathologyRequest):
    if req.Biopsy_malignant:
        outcome = "malignant"
        note = "Confirmed malignant on biopsy"
    else:
        outcome = "benign"
        note = "Confirmed benign on biopsy"
    return PathologyResponse(Case_id=req.Case_id, pathology=outcome, note=note)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pathologist_agent:app", host="0.0.0.0", port=8002, reload=True)
