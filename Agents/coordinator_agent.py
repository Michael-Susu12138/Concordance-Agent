from fastapi import FastAPI, HTTPException
import httpx

app = FastAPI()

MCP_URL    = "http://localhost:8000"
RAD_URL    = "http://localhost:8001"
PATH_URL   = "http://localhost:8002"
CONC_URL   = "http://localhost:8003"
NOTIF_URL  = "http://localhost:8004"

@app.post("/process_case/{case_id}")
async def process_case(case_id: int):
    async with httpx.AsyncClient() as client:
        # 1) get patient context
        r = await client.get(f"{MCP_URL}/mcp/patient/{case_id}")
        if r.status_code != 200:
            raise HTTPException(status_code=404, detail="Case not found in MCP")
        case_data = r.json()

        # 2) radiology analysis
        rad_req = await client.post(
            f"{RAD_URL}/analyze_radiology", json={
                "Case_id": case_id,
                "TI_RADS": case_data["TI_RADS"]
            }
        )
        rad_res = rad_req.json()

        # 3) pathology analysis
        path_req = await client.post(
            f"{PATH_URL}/analyze_pathology", json={
                "Case_id": case_id,
                "Biopsy_benign": case_data["Biopsy_benign"],
                "Biopsy_malignant": case_data["Biopsy_malignant"]
            }
        )
        path_res = path_req.json()

        # 4) concordance-discordance analysis
        conc_req = await client.post(
            f"{CONC_URL}/evaluate_concordance",
            json={"radiology": rad_res, "pathology": path_res}
        )
        conc_res = conc_req.json()

        # 5) notify relevant personel
        notif_req = await client.post(
            f"{NOTIF_URL}/notify",
            json={"case_id": case_id, "concordance": conc_res}
        )
        notif_res = notif_req.json()

    return {
        "case_id": case_id,
        "radiology": rad_res,
        "pathology": path_res,
        "concordance": conc_res,
        "notification": notif_res
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("coordinator_agent:app", host="0.0.0.0", port=8005, reload=True)
