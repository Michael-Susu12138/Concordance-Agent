from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class RadiologyResult(BaseModel):
    Case_id: int
    tr_level: int
    interpretation: str

class PathologyResult(BaseModel):
    Case_id: int
    pathology: str
    note: str

class ConcordanceRequest(BaseModel):
    radiology: RadiologyResult
    pathology: PathologyResult

class ConcordanceResponse(BaseModel):
    concordant: bool
    discrepancy_type: Optional[str]
    explanation: str

@app.post("/evaluate_concordance", response_model=ConcordanceResponse)
async def evaluate_concordance(req: ConcordanceRequest):
    level = req.radiology.tr_level 
    path = req.pathology.pathology.lower()

    if level >= 3 and path == "benign":
        return ConcordanceResponse(
            concordant=False,
            discrepancy_type="false_positive",
            explanation=(
                f"TI-RADS level {level} indicated moderate/high suspicion, "
                "but pathology returned benign."
            )
        )

    if level <= 2 and path == "malignant":
        return ConcordanceResponse(
            concordant=False,
            discrepancy_type="false_negative",
            explanation=(
                f"TI-RADS level {level} indicated benign/not suspicious, "
                "but pathology returned malignant."
            )
        )

    return ConcordanceResponse(
        concordant=True,
        discrepancy_type=None,
        explanation="Imaging and pathology findings are concordant."
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("concordance_agent:app", host="0.0.0.0", port=8003, reload=True)
