import os, json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
# synthetic patient data
BASE_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(BASE_DIR, "concord_synthetic_dataset.json")

# load data once at startup
with open(DATA_FILE, "r", encoding="utf-8") as f:
    DATA = {case["Case_id"]: case for case in json.load(f)}

class PatientContext(BaseModel):
    Case_id: int
    TI_RADS: int
    Biopsy_benign: int
    Biopsy_malignant: int
    Concordant: int
    Discordant: int

@app.get("/mcp/patient/{case_id}", response_model=PatientContext)
async def get_patient(case_id: int):
    case = DATA.get(case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    # map JSON keys to Python identifiers
    return {
        "Case_id": case["Case_id"],
        "TI_RADS": case["TI-RADS"],
        "Biopsy_benign": case["Biopsy_benign"],
        "Biopsy_malignant": case["Biopsy_malignant"],
        "Concordant": case["Concordant"],
        "Discordant": case["Discordant"],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8000, reload=True)
