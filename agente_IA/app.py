
from smolagents import ToolCallingAgent, tool, DuckDuckGoSearchTool
from dotenv import load_dotenv
from smolagents.models import OpenAIServerModel

load_dotenv()
#HERRAMIENTA DE PRUEBA
@tool
def send_message_to(destinatario: str, mensaje: str) -> str:
    """
    Útil para enviar un mensaje de correo electrónico a un destinatario.
    
    Args:
        destinatario: La persona que recibe el email.
        mensaje: El contenido del correo.
    """ 
    print(f"\n[ACCION] Enviando email a {destinatario}...")
    return f"¡Éxito! El mensaje '{mensaje}' ha sido enviado a {destinatario}."

#MODELO DE PRUEBA ANTES DE USAR OLLAMA
model_id = "meta-llama/Llama-3.1-8B-Instruct"

llm=OpenAIServerModel(
    model_id="mistral",
    api_base="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0,
    response_format={"type":"json.Object"}
)

agente = ToolCallingAgent(
    model=llm,
    tools=[send_message_to, DuckDuckGoSearchTool()],
    max_steps=4
)


print("\n--- Iniciando Agente ---")
response = agente.run("DIME EL TIEMPO EN Andujar mañana a las 7:00")
print("\n--- Respuesta Final ---")
print(response)