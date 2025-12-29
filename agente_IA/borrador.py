# Configuración del Modelo
llm = OpenAIServerModel(
    model_id="gpt-oss:20b-cloud",
    #model_id="mistral",
    api_base="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0
)



# Templates
planning_template = PlanningPromptTemplate(
    initial_plan="Identifica los pasos iniciales.",
    plan="Planea qué herramientas usar.",
    update_plan_pre_messages="Revisa pasos anteriores.",
    update_plan_post_messages="Actualiza según hallazgos."
)

prompt_templates = PromptTemplates(
 system_prompt="""Eres un experto en el IES Jándula. Tu tarea es responder sobre módulos y FP.

        DEBES RESPONDER SIEMPRE con un ÚNICO objeto JSON.
        NO añadas explicaciones, texto adicional, introducciones ni mensajes fuera del JSON.
        NO uses backticks.
        NO uses bloques de código.

        FORMATO OBLIGATORIO PARA LLAMADAS A HERRAMIENTAS:
        {
        "name": "nombre_de_la_herramienta",
        "arguments": {
            "arg1": "valor"
        }
        }

        FORMATO OBLIGATORIO PARA RESPUESTA FINAL (sin herramientas):
        {
        "name": "final_answer",
        "arguments": {
            
        }

        Si necesitas usar varias herramientas, respóndelas una por una, nunca juntas.
    """,
    planning=planning_template,
    #managed_agent=ManagedAgentPromptTemplate(task="Usa herramientas paso a paso.", report="Resume hallazgos."),
    #final_answer=FinalAnswerPromptTemplate(pre_messages="Considera todo antes de responder.", post_messages="Responde en español claro.")
)

#Instancia del Agente
""" agente = ToolCallingAgent(
    model=llm,
    tools=[ guia_profesorado],
    max_steps=5,
    prompt_templates=prompt_templates
) """