from smolagents import ToolCallingAgent, OpenAIServerModel, PromptTemplates, PlanningPromptTemplate, ManagedAgentPromptTemplate, FinalAnswerPromptTemplate
from tools_config import dar_respuesta_final, tool_buscar_en_web, guia_profesorado
import asyncio
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain import tools
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages 
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition 
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
import requests
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage

#_________________________________________________________________________________________________________________
load_dotenv()

# Obtenemos la clave
api_key = os.getenv("TAVILY_API_KEY")

# Verificamos si la clave existe antes de intentar asignarla
if api_key is None:
    print("⚠️ Error: No se encontró TAVILY_API_KEY en el archivo .env")
    # Opcional: puedes poner la clave a mano aquí temporalmente para probar
    # api_key = "tvly-tu_clave_real" 
else:
    os.environ["TAVILY_API_KEY"] = api_key
#_________________________________________________________________________________________________________________
class Estado(TypedDict):
    messages: Annotated[list, add_messages]

def _ejecutar_herramientas_sincro(estado: Estado, herramientas: list):
    """Ejecuta herramientas de forma sincrónica sin ThreadPoolExecutor."""
    mensajes = estado["messages"]
    ultimo_mensaje = mensajes[-1]
    
    # Si el último mensaje no contiene tool_calls, no hay nada que hacer
    if not hasattr(ultimo_mensaje, 'tool_calls') or not ultimo_mensaje.tool_calls:
        return {"messages": []}
    
    herramientas_por_nombre = {h.name: h for h in herramientas}
    nuevos_mensajes = []
    
    # Ejecutar cada herramienta de forma sincrónica
    for tool_call in ultimo_mensaje.tool_calls:
        herramienta = herramientas_por_nombre.get(tool_call["name"])
        if not herramienta:
            resultado = f"Herramienta '{tool_call['name']}' no encontrada"
        else:
            try:
                resultado = herramienta.invoke(tool_call["args"])
            except Exception as e:
                resultado = f"Error ejecutando {tool_call['name']}: {str(e)}"
        
        nuevos_mensajes.append(
            ToolMessage(
                content=str(resultado),
                tool_call_id=tool_call["id"],
                name=tool_call["name"]
            )
        )
    
    return {"messages": nuevos_mensajes}

def inicializar_agente():
    """Inicializa el agente de forma sincrónica."""
    from playwright.sync_api import sync_playwright
    
    # Usar navegador sincrónico (PlayWrightBrowserToolkit requiere sync, no async)
    playwright = sync_playwright().start()
    navegador = playwright.chromium.launch(headless=True)
    page = navegador.new_page()
    page.goto("https://blogsaverroes.juntadeandalucia.es/iesjandula/", timeout=15000)

    conjunto_herramientas = PlayWrightBrowserToolkit(sync_browser=navegador)
    tool_busqueda_general = TavilySearch(
        max_results=3, 
        tavily_api_key=api_key
    )
    herramientas_navegador = conjunto_herramientas.get_tools()
    SYSTEM_PROMPT = """Eres el Asistente Oficial del IES Jándula (Andújar). 
        Tu ámbito de actuación es EXCLUSIVAMENTE el centro educativo IES Jándula.
        Respondes SIEMPRE en español mientras no te pidan que respondas en otro idioma.

        PRIORIDAD DE BÚSQUEDA:
        1. Información Interna: Usa 'guia_profesorado' para datos sobre profesores, guía de actuación, orientación de profesorado... o navega en la
            Web Oficial: Para noticias, contacto, calendarios y eventos, oferta de módulos, blogs, oferta educativa navega en https://blogsaverroes.juntadeandalucia.es/iesjandula/ usando Playwright.
        3. Internet (Tavily): Úsalo ÚNICAMENTE si el usuario pregunta algo general necesario para entender un concepto del centro, o si buscas una noticia externa que mencione específicamente al 'IES Jándula'.

        REGLA CRÍTICA: 
        - No des consejos generales de educación en Andalucía a menos que estén publicados en la web del centro. 
        - Si la información no es específica del IES Jándula, responde: "Esa información no consta en los registros oficiales del IES Jándula".
        - Responde siempre de forma concisa y en español.
        """
    chat = ChatOllama(model="gpt-oss:20b-cloud", temperature=0, SystemMessage=SYSTEM_PROMPT)
    llm_herramientas = chat.bind_tools([guia_profesorado, dar_respuesta_final, tool_busqueda_general]+ herramientas_navegador)

    
    
    def chatbot(estado: Estado):
        try:
            return {"messages": [llm_herramientas.invoke(estado["messages"])]}
        except Exception as e:
            print(e)
            return {"messages": [("assistant", "Lo siento, he tenido un problema técnico buscando esa información.")]}
    
    # Nodo de herramientas sincrónico
    def herramientas_nodo(estado: Estado):
        return _ejecutar_herramientas_sincro(estado, herramientas_navegador+[guia_profesorado,dar_respuesta_final, tool_busqueda_general])

    constructor_grafo = StateGraph(Estado)
    constructor_grafo.add_node("chatbot",chatbot)
    constructor_grafo.add_node("tools", herramientas_nodo)
    constructor_grafo.add_conditional_edges("chatbot",tools_condition)
    constructor_grafo.add_edge("tools","chatbot")
    constructor_grafo.add_edge(START,"chatbot")
    


    memoria = MemorySaver()
    grafo=constructor_grafo.compile(checkpointer=memoria)
    return grafo

async def main_async():
    """Funcion ejecutar aplicacion del agente de forma asincrona"""
    print("Inicializando agente...")
    grafo= await inicializar_agente()
    print("Agente inicializado.")

    #interfaz Gradio
    import gradio as gr
    async def chat_wrapper_async(entrada_usuario: str, historial):
        import uuid
        session_id=uuid.uuid4()
        configuracion={"configurable": {"thread_id": "10"},"recursion_limit": 30}
        resultado= await grafo.ainvoke(
            {"messages": [{"role": "user", "content": entrada_usuario}]},
            config=configuracion
        )
        return resultado["messages"][-1].content
    
    print("Iniciando interfaz Gradio...")
    demo = gr.ChatInterface(
        chat_wrapper_async,
        title="Agente IES Jándula",
        description="Agente conversacional especializado en el IES Jándula."
    )

    demo.launch(prevent_thread_lock=True)

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Cerrando agente...")

if __name__ == "__main__":
    asyncio.run(main_async())