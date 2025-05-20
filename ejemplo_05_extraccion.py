# indexador_faq.py
import re, json
import pandas as pd
import numpy as np
from openai import OpenAI

import config

client = OpenAI(project=config.OPENAI_PROJECT, api_key=config.OPENAI_API_KEY)

def cargar_faq_desde_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        contenido = f.read()
    bloques = re.findall(r"### (.*?)\n+([^\n#]+(?:\n(?!###).+)*)", contenido)
    return [{"pregunta": p.strip(), "respuesta": r.strip()} for p, r in bloques]

def get_embedding(texto):
    texto = texto.replace("\n", " ")
    return client.embeddings.create(input=[texto], model="text-embedding-ada-002").data[0].embedding

if __name__ == "__main__":
    faq = cargar_faq_desde_markdown("docs/procesos_academicos.md")
    for item in faq:
        item["contenido"] = item["pregunta"] + " " + item["respuesta"]
        item["embedding"] = get_embedding(item["contenido"])

    df = pd.DataFrame(faq)
    df.to_csv("docs/faq_embed.csv", index=False)
    print("✅ Base embebida guardada en 'docs/faq_embed.csv'")
