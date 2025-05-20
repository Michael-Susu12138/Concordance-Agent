from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class NotifyRequest(BaseModel):
    case_id: int
    concordance: dict

class NotifyResponse(BaseModel):
    status: str
    message: str

@app.post("/notify", response_model=NotifyResponse)
async def notify(req: NotifyRequest):
    conc = req.concordance
    if not conc.get("concordant", True):
        msg = f"Discordant Case {req.case_id}: {conc['explanation']}"
        status = "alert_sent"
    else:
        msg = f"Concordant Case {req.case_id}: no action needed."
        status = "logged"
    # could send email/HL7? wait for Bora's idea
    print(msg)
    return NotifyResponse(status=status, message=msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("notification_agent:app", host="0.0.0.0", port=8004, reload=True)
