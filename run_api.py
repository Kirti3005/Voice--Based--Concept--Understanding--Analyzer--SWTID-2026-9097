# VBCUA/run_api.py
import uvicorn

if __name__ == "__main__":
    print("Starting VBCUA REST API server...")
    uvicorn.run("backend.api:app", host="127.0.0.1", port=8000, reload=True)
