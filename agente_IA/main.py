
import os
from fastapi.concurrency import run_in_threadpool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from smolagents import ToolCallingAgent, tool, DuckDuckGoSearchTool,ApiWebSearchTool
from dotenv import load_dotenv
from smolagents.models import OpenAIServerModel
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from smolagents import AgentMemory, PromptTemplates, PlanningPromptTemplate, ManagedAgentPromptTemplate, FinalAnswerPromptTemplate

from playwright.sync_api import sync_playwright
from urllib.parse import urlparse, urljoin, urldefrag
import re
from rapidfuzz import fuzz

from fastapi import FastAPI
from pydantic import BaseModel

class Pregunta(BaseModel):
    pregunta: str

load_dotenv()

app = FastAPI()
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

#_____________________________________________________________________________________________________________________
def clean_text(html):
    """Elimina scripts, estilos, HTML y normaliza el texto."""
    html = re.sub(r"<(script|style).*?>.*?</\1>", "", html, flags=re.DOTALL)
    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"\s+", " ", html)
    return html.strip()


def is_internal_link(link, base_domain):
    parsed = urlparse(link)
    return (parsed.netloc == "" or parsed.netloc == base_domain) and parsed.scheme in ["http", "https", ""]


def normalize_url(url):
    url = urldefrag(url).url
    if url.endswith("/"):
        url = url[:-1]
    return url


def crawl_site(base_url, max_pages=200):
    visited = set()
    to_visit = [normalize_url(base_url)]
    index = {}

    base_domain = urlparse(base_url).netloc

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()

        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            url = normalize_url(url)

            if url in visited:
                continue

            try:
                print(f"Navegando en: {url}")
                page.goto(url, timeout=15000)
                page.wait_for_load_state("domcontentloaded")

                #Extrae solo el MAIN , para evitar el error de las cookies que antes tenía
                main_html = page.eval_on_selector("main", "el => el.innerHTML")  
                if not main_html:  
                    #SI no hay main, no queda otra que devolver el body
                    main_html = page.eval_on_selector("body", "el => el.innerHTML")

                text = clean_text(main_html)
                index[url] = text
                visited.add(url)

                #Esto es para sacar los enlaces internos de la web
                links = page.eval_on_selector_all("a", "els => els.map(e => e.href)")
                links = [l for l in links if l.startswith("http")]

                for link in links:
                    abs_url = normalize_url(urljoin(base_url, link))
                    if is_internal_link(abs_url, base_domain):
                        if abs_url not in visited and abs_url not in to_visit:
                            to_visit.append(abs_url)

            except Exception as e:
                print(f"Error en {url}: {e}")
                continue

        browser.close()

    return index


def search(query, index, threshold=30):
    query = query.lower()
    results = []

    for url, text in index.items():
        score = fuzz.partial_ratio(query, text.lower())
        if score >= threshold:
            pos = text.lower().find(query.split()[0])
            snippet = text[max(0, pos - 2000) : pos + 2000] if pos != -1 else text[:5000]
            results.append((score, url, snippet))

    results.sort(key=lambda x: x[0], reverse=True)
    return [(url, snippet) for score, url, snippet in results]
#_____________________________________________________________________________________________________________________
@tool
def tool_buscar_en_web(query: str) -> str:
    """
    Útil para obtener información sobre oferta educativa de módulos, FP, grado medio, grado superior,
    información sobre cada módulo, asignaturas, contactos,etc.
    Busca la página más relevante dentro de https://fp.iesjandula.es/
    usando rastreo + búsqueda difusa.
    Devuelve SOLO la mejor URL.
    Args:
        query (str): Término o frase que se desea buscar dentro del sitio web.

    Returns:
        str: Texto formateado con cada URL encontrada y su fragmento relevante. 
             Si no hay coincidencias, devuelve “No se encontraron resultados.”
    """
    base_url = "https://fp.iesjandula.es/"
    index = crawl_site(base_url, max_pages=80)

    results = search(query, index)

    if not results:
        return "No se encontraron resultados."

    # Solo devolvemos la mejor URL
    best_url, snippet = results[0]

    return f"URL encontrada: {best_url}\nContenido relevante: {snippet}"

#___________________________________________________________________________________________________________________
#TOOL PARA CONSULTAR LA GUÍA DEL PROFESORADO

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_db_path = os.path.join(os.getcwd(), "data", "chroma_db")

vector_store = Chroma(
    collection_name="guia_profesorado",
    embedding_function=embeddings,
    persist_directory=persist_db_path
)

@tool
def guia_profesorado(search:str)->str:
    """Consulta la guía oficial del profesorado del IES Jándula para el curso 2025/26.
    
    Úsala SIEMPRE que el usuario tenga dudas sobre:
    - Organización del Centro: Datos de contacto, plano de las instalaciones, cargos directivos y jefaturas de departamento.
    - Horarios y Calendario: Jornada laboral (horas lectivas y no lectivas), calendario escolar, fechas de evaluaciones, claustros y reuniones de departamento.
    - Gestión de Ausencias y Bajas: Protocolo para notificar faltas vía Séneca, licencias por enfermedad (Anexo I) y cambios de clase.
    - Normativa y Funciones: Obligaciones del profesor de guardia (patio, pasillos, aula de convivencia), atención a familias y protocolos de accidentes del alumnado.
    - Actividades y Programación: Normas para actividades extraescolares, viajes de alumnos, uso de la biblioteca, reserva de aulas específicas y contenidos de las programaciones (ESO, Bachillerato, FP).
    - Planes Educativos: Información sobre proyectos como Erasmus+, Bilingüismo, Escuela Espacio de Paz, Proyectos Internacionales, etc.
    
    Args:
        search (str): Término de búsqueda, pregunta o consulta específica sobre la normativa u organización del centro.
        
    Returns:
        str: Información relevante extraída directamente de la guía oficial.
    """
    #los K son los fragmentos
    docs = vector_store.similarity_search(search, k=10)
    texto_sucio = "\n\n".join([doc.page_content for doc in docs])
    texto_limpio = " ".join(texto_sucio.split())
    return texto_limpio
#_____________________________________________________________________________________________________________________

#MODELO DE PRUEBA ANTES DE USAR OLLAMA
model_id = "meta-llama/Llama-3.1-8B-Instruct"

llm=OpenAIServerModel(
    model_id="gpt-oss:20b-cloud",
    api_base="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0,
    #tool_choice="auto",
    #response_format={"type":"json.Object"}
)
web_search_tool = ApiWebSearchTool(rate_limit=50.0)

memory = AgentMemory(system_prompt="""
Eres un experto en el IES Jándula.
Tu objetivo es responder preguntas sobre módulos, ciclos, asignaturas y FP.
Usa las herramientas disponibles para buscar información en la web oficial.
Responde siempre en español de manera clara y concisa.
No inventes información.
""")
planning_template = PlanningPromptTemplate(
    initial_plan="Identifica los pasos iniciales para recopilar la información.",
    plan="Planea qué herramientas usar para obtener información precisa sobre el IES Jándula.",
    update_plan_pre_messages="Revisa los pasos anteriores antes de actualizar el plan.",
    update_plan_post_messages="Actualiza tu plan según los hallazgos más recientes."
)


managed_template = ManagedAgentPromptTemplate(
    task="Usa las herramientas disponibles paso a paso para recopilar la información solicitada.",
    report="Resume los hallazgos relevantes de cada paso."
)

final_answer_template = FinalAnswerPromptTemplate(
    pre_messages="Considera toda la información recopilada antes de dar la respuesta final.",
    post_messages="Responde en español, de manera clara y resumida."
)

#MUYYYYY IMPORTANTE escribir el formato de respuesta, si no, toolCallingAgent tiene problemas
prompt_templates = PromptTemplates(
    system_prompt=""""
        Eres un experto en el IES Jándula.
        Tu objetivo es responder preguntas sobre módulos, ciclos, contactos, asignaturas y FP.
        Usa las herramientas disponibles para buscar información.
        Si el usuario pregunta por horarios, normativa, profesores o jornada laboral, DEBES usar primero 'guia_profesorado'.
        Solo si la información no aparece ahí, utiliza 'tool_buscar_en_web' para consultar la web de FP.
        Cuando el contenido incluya varios ciclos, debes enumerarlos TODOS en lugar de seleccionar uno.
        Responde siempre en español de manera clara y concisa.
        No inventes información.
        Tu formato de respuesta debe ser:
            {
            "name": "nombre_de_la_herramienta",
            "arguments": {"arg_name": "valor"}
            }
        SOLO puedes responder de dos formas:
        1. Llamando a una herramienta con un ÚNICO bloque JSON.
        2. Dando la respuesta final prefijada con 'Final Answer:'.

        Formato estricto de herramienta:
        {"name": "nombre_de_la_herramienta", "arguments": {"arg_name": "valor"}}

        No añadas texto antes ni después del JSON. No llames a varias herramientas a la vez.
        """,
    planning=planning_template,
    managed_agent=managed_template,
    final_answer=final_answer_template
)

agente = ToolCallingAgent(
    model=llm,
    tools=[tool_buscar_en_web, guia_profesorado],
    max_steps=5,
    prompt_templates=prompt_templates
   
)

print("\n--- Iniciando Agente ---")

# Expresiones que hacen salir del programa
exit_commands = {"salir", "exit", "quit", "adios", "adiós", "bye", "chao", "hasta luego"}

""" while True:
    request = input("tú: ").strip().lower()

    if request in exit_commands:
        print("\nIES Jándula: ¡Ha sido un placer ayudarte! Si necesitas más asistencia, aquí estaré.")
        break

    if not request:
        continue

    response = agente.run(request)

    print("\n--- Respuesta Final ---")
    print("\nIES Jándula:",response)
    print() """
#_____________________________________________________________________________________________________________________

@app.post("/consulta/")
async def consulta_ies_jandula(data: Pregunta):
    try:
        print(f"Recibida pregunta: {data.pregunta}")
        
        #RUN IN THREADPOOL ES PARA EVITAR CONFLICTO CON SYNC PLAYWRIGHT
        respuesta = await run_in_threadpool(agente.run, data.pregunta)
        
        return {"respuesta": respuesta}
    except Exception as e:
        print(f"Error detectado: {e}")
        return {"error": str(e)}
