import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv
from agente import agente

load_dotenv()

app = FastAPI(title="Agente IES Jándula API")

class Pregunta(BaseModel):
    pregunta: str

@app.post("/consulta/")
async def consulta_ies_jandula(data: Pregunta):
    try:
        print(f"Recibida pregunta: {data.pregunta}")
        # run_in_threadpool evita el error de Playwright Sync en el loop de FastAPI
        respuesta = await run_in_threadpool(agente.run, data.pregunta)
        return {"respuesta": respuesta}
    except Exception as e:
        print(f"Error en API: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)