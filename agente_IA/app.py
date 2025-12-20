from dotenv import load_dotenv
import os
from os import getenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import create_agent
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
#device = 0 if torch.cuda.is_available() else -1
#print("Usando:", "GPU" if device == 0 else "CPU")

model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,    # usar bf16
    device_map="auto",             # usar GPU si existe
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    do_sample=False,
)
hf = HuggingFacePipeline(pipeline=pipe)

""" hf = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 10},
) """

chat_model = ChatHuggingFace(llm=hf)
print(chat_model.invoke("hola mi niño"))
