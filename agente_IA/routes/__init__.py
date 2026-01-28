"""
Paquete de rutas para la API.
"""
from .agent_routes import router, inicializar_agente_app

__all__ = ["router", "inicializar_agente_app"]
