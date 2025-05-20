from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class RadiologyRequest(BaseModel):
    Case_id: int
    TI_RADS: int = Field(..., description="Raw TI-RADS point score (1,2,3,4–6,7+)")

class RadiologyResponse(BaseModel):
    Case_id: int
    tr_level: int = Field(..., description="Mapped TI-RADS level (1–5)")
    interpretation: str

@app.post("/analyze_radiology", response_model=RadiologyResponse)
async def analyze_radiology(req: RadiologyRequest):
    pts = req.TI_RADS

    # raw point score to TI-RADS level
    if pts == 1:
        level = 1
        interp = "Benign appearance"
    elif pts == 2:
        level = 2
        interp = "Probably benign"
    elif pts == 3:
        level = 3
        interp = "Mildly suspicious"
    elif 4 <= pts <= 6:
        level = 4
        interp = "Moderately suspicious"
    elif pts >= 7:
        level = 5
        interp = "Highly suspicious of malignancy"
    else:
        interp = "unknown"

    return RadiologyResponse(
        Case_id=req.Case_id,
        tr_level=level,
        interpretation=interp
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("radiologist_agent:app", host="0.0.0.0", port=8001, reload=True)
