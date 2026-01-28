"""
Rutas para consultar al agente de IES Jándula.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio
import threading
import queue
from agente import inicializar_agente

# Router para las rutas del agente
router = APIRouter(prefix="/agente", tags=["Agente"])

# Modelos de solicitud/respuesta
class ConsultaRequest(BaseModel):
    """Modelo para las consultas al agente."""
    pregunta: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "pregunta": "¿Cuál es el horario del IES Jándula?"
            }
        }

class ConsultaResponse(BaseModel):
    """Modelo para la respuesta del agente."""
    respuesta: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "respuesta": "El horario del IES Jándula es de 8:00 a 14:30..."
            }
        }

# Variables globales para el worker thread del agente
_agente_grafo = None
_agente_queue = queue.Queue()
_agente_response_queue = queue.Queue()
_worker_thread = None
_worker_started = threading.Event()
_worker_ready = False

def _agente_worker():
    """Worker thread que ejecuta todas las operaciones del agente."""
    global _agente_grafo, _worker_ready
    
    # Crear event loop propio para este thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    print("🚀 Inicializando agente...")
    try:
        _agente_grafo = inicializar_agente()
        print("✅ Agente inicializado correctamente.")
        _worker_ready = True
        _worker_started.set()
    except Exception as e:
        print(f"❌ Error al inicializar agente: {e}")
        _agente_response_queue.put(("error", str(e)))
        _worker_started.set()
        loop.close()
        return
    
    # Loop que procesa consultas
    while True:
        try:
            # Usar timeout pequeño para permitir que el thread se cierre si es necesario
            item = _agente_queue.get(timeout=0.5)
            
            if item is None:  # Señal de parada
                break
            
            pregunta = item
            try:
                print(f"[API] Recibida consulta: {pregunta}")
                
                # Ejecutar la consulta con max_concurrency=1 para evitar ThreadPoolExecutor
                config = {
                    "configurable": {"thread_id": "default"},
                    "max_concurrency": 1
                }
                respuesta = _agente_grafo.invoke(
                    {"messages": [("user", pregunta)]},
                    config
                )
                
                # Extraer la respuesta
                mensajes = respuesta.get("messages", [])
                if mensajes:
                    ultimo_mensaje = mensajes[-1]
                    if isinstance(ultimo_mensaje, tuple):
                        respuesta_texto = ultimo_mensaje[1]
                    else:
                        respuesta_texto = str(ultimo_mensaje.content) if hasattr(ultimo_mensaje, 'content') else str(ultimo_mensaje)
                else:
                    respuesta_texto = "No se obtuvo respuesta del agente"
                
                print(f"[API] Respuesta generada: {respuesta_texto}")
                _agente_response_queue.put(("success", respuesta_texto))
                
            except Exception as e:
                print(f"[ERROR] Error en consulta al agente: {str(e)}")
                import traceback
                traceback.print_exc()
                _agente_response_queue.put(("error", str(e)))
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[ERROR] Error en worker thread: {str(e)}")
            break
    
    loop.close()

async def inicializar_agente_app():
    """Inicia el worker thread al arrancar la app."""
    global _worker_thread, _worker_ready
    
    if _worker_thread is None:
        _worker_thread = threading.Thread(target=_agente_worker, daemon=False)
        _worker_thread.start()
        
        # Esperar a que el agente se inicialice
        _worker_started.wait(timeout=120)
        
        if not _worker_ready:
            # Verificar si hubo error durante la inicialización
            if not _agente_response_queue.empty():
                status, msg = _agente_response_queue.get()
                if status == "error":
                    raise RuntimeError(f"Error inicializando agente: {msg}")
            raise RuntimeError("El agente tardó demasiado en inicializarse")

@router.post("/consulta", response_model=ConsultaResponse)
async def consultar_agente(consulta: ConsultaRequest):
    """
    Consulta al agente del IES Jándula.
    
    Args:
        consulta: Objeto con la pregunta para el agente
        
    Returns:
        Respuesta del agente en formato JSON
        
    Raises:
        HTTPException: Si hay un error al procesar la consulta
    """
    try:
        if not consulta.pregunta.strip():
            raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")
        
        # Enviar la consulta al worker thread a través de la queue
        _agente_queue.put(consulta.pregunta)
        
        # Esperar la respuesta (con timeout)
        status, resultado = _agente_response_queue.get(timeout=120)
        
        if status == "error":
            raise HTTPException(status_code=500, detail=f"Error al procesar la consulta: {resultado}")
        
        return ConsultaResponse(respuesta=resultado)
        
    except HTTPException:
        raise
    except queue.Empty:
        raise HTTPException(status_code=504, detail="Timeout esperando respuesta del agente")
    except Exception as e:
        print(f"[ERROR] Error en consulta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la consulta: {str(e)}")

@router.get("/health")
async def health_check():
    """Verifica que el servicio está activo."""
    return {"status": "ok", "servicio": "Agente IES Jándula"}
