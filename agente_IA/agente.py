from smolagents import ToolCallingAgent, OpenAIServerModel, PromptTemplates, PlanningPromptTemplate, ManagedAgentPromptTemplate, FinalAnswerPromptTemplate
from tools_config import dar_respuesta_final, tool_buscar_en_web, guia_profesorado
import asyncio
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain import tools
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages 
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition 
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
import requests

#_________________________________________________________________________________________________________________




#_________________________________________________________________________________________________________________

class Estado(TypedDict):
    messages: Annotated[list, add_messages]

async def inicializar_agente():
    from playwright.async_api import async_playwright
    playwright = await async_playwright().start()
    navegador_async = await playwright.chromium.launch(headless=True)
    page = await navegador_async.new_page()
    await page.goto("https://fp.iesjandula.es/", timeout=15000)

    conjunto_herramientas = PlayWrightBrowserToolkit(async_browser=navegador_async)
    herramientas_navegador = conjunto_herramientas.get_tools()
    #herramientas_navegador = asyncio.run(inicializar_agente())
    SYSTEM_PROMPT = """Eres un asistente experto del IES Jándula (Andújar). 
    Tu objetivo es ayudar a alumnos y profesores con información sobre módulos, ciclos formativos y noticias del centro.

    REGLAS DE NAVEGACIÓN:
    1. Para cualquier consulta externa, utiliza SIEMPRE como punto de partida la web: https://iesjandula.es/
    2. Si necesitas buscar noticias o calendarios, navega primero por las secciones de esa web.
    3. Responde de forma clara, educada y en español.
    4. Si no encuentras la información en la web oficial ni en la guía del profesorado, admítelo.

    Intenta encontrar la información en el menor número de clics posible
    """
    chat = ChatOllama(model="gpt-oss:20b-cloud", temperature=0, SystemMessage=SYSTEM_PROMPT)
    llm_herramientas = chat.bind_tools([guia_profesorado, dar_respuesta_final]+ herramientas_navegador)

    
    
    def chatbot(estado: Estado):
        try:
            return {"messages": [llm_herramientas.invoke(estado["messages"])]}
        except Exception as e:
            return {"messages": ["No se pudo obtener la información"]}

    constructor_grafo = StateGraph(Estado)
    constructor_grafo.add_node("chatbot",chatbot)
    constructor_grafo.add_node("tools", ToolNode(tools=herramientas_navegador+[guia_profesorado,dar_respuesta_final]))
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