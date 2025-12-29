import os
import re
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse, urljoin, urldefrag
from rapidfuzz import fuzz
#from smolagents import tool
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- UTILIDADES DE SCRAPING ---
def clean_text(html):
    html = re.sub(r"<(script|style).*?>.*?</\1>", "", html, flags=re.DOTALL)
    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"\s+", " ", html)
    return html.strip()

def is_internal_link(link, base_domain):
    parsed = urlparse(link)
    return (parsed.netloc == "" or parsed.netloc == base_domain) and parsed.scheme in ["http", "https", ""]

def normalize_url(url):
    url = urldefrag(url).url
    if url.endswith("/"): url = url[:-1]
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
            url = normalize_url(to_visit.pop(0))
            if url in visited: continue
            try:
                print(f"Navegando en: {url}")
                page.goto(url, timeout=15000)
                page.wait_for_load_state("domcontentloaded")
                main_html = page.eval_on_selector("main", "el => el.innerHTML") or page.eval_on_selector("body", "el => el.innerHTML")
                index[url] = clean_text(main_html)
                visited.add(url)
                links = page.eval_on_selector_all("a", "els => els.map(e => e.href)")
                for link in [l for l in links if l.startswith("http")]:
                    abs_url = normalize_url(urljoin(base_url, link))
                    if is_internal_link(abs_url, base_domain) and abs_url not in visited:
                        to_visit.append(abs_url)
            except Exception: continue
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

# --- HERRAMIENTAS (@tool) ---
@tool
def tool_buscar_en_web(query: str) -> str:
    """Busca información de oferta educativa en https://fp.iesjandula.es/
    Realiza una búsqueda en el sitio web del IES Jándula y devuelve la URL y un fragmento relevante.

    Args:
        query (str): La consulta de búsqueda.

    Returns: 
        str: La URL y el fragmento relevante encontrado.

    """
    base_url = "https://fp.iesjandula.es/"
    index = crawl_site(base_url, max_pages=80)
    results = search(query, index)
    if not results: return "No se encontraron resultados."
    best_url, snippet = results[0]
    return f"URL encontrada: {best_url}\nContenido relevante: {snippet}"

#Configuración Vector DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_db_path = os.path.join(os.getcwd(), "data", "chroma_db")
vector_store = Chroma(collection_name="guia_profesorado", embedding_function=embeddings, persist_directory=persist_db_path)

@tool
def guia_profesorado(search: str) -> str:
    """Consulta la guía oficial del profesorado del IES Jándula 2025/26.

    Args:
        search (str): La consulta de búsqueda.

    Returns: 
        str: La información relevante encontrada en la guía.
    """
    docs = vector_store.similarity_search(search, k=10)
    return " ".join(("\n\n".join([doc.page_content for doc in docs])).split())

@tool
def dar_respuesta_final(respuesta: str):
    """Usa esta herramienta cuando tengas la información necesaria para responder al usuario y quieras finalizar la sesión."""
    return respuesta