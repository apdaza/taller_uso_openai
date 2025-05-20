import os
import re
import json
import numpy as np
from openai import OpenAI
import config

client = OpenAI(project=config.OPENAI_PROJECT, api_key=config.OPENAI_API_KEY)

# -----------------------------
# Paso 1: Cargar y procesar el archivo Markdown
# -----------------------------
def cargar_preguntas_desde_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        contenido = file.read()
    
    bloques = re.findall(r"### (.*?)\n+([^\n#]+(?:\n(?!###).+)*)", contenido)
    faq = [{"pregunta": p.strip(), "respuesta": r.strip()} for p, r in bloques]
    return faq

# -----------------------------
# Paso 2: Calcular embeddings con OpenAI
# -----------------------------
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# -----------------------------
# Paso 3: Buscar por similitud de coseno
# -----------------------------
def similitud_coseno(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# -----------------------------
# Paso 4: Procesamiento principal
# -----------------------------
def buscar_respuesta(faq, pregunta_usuario):
    print("Calculando embeddings...")
    vector_usuario = np.array(get_embedding(pregunta_usuario))

    for item in faq:
        item["embedding"] = np.array(get_embedding(item["pregunta"] + " " + item["respuesta"]))
        item["similitud"] = similitud_coseno(item["embedding"], vector_usuario)

    mejor = max(faq, key=lambda x: x["similitud"])
    
    print(f"\n📚 Pregunta más parecida:\n{mejor['pregunta']}")
    print(f"💡 Respuesta sugerida:\n{mejor['respuesta']}")
    print(f"🔢 Similitud: {mejor['similitud']:.4f}")

# -----------------------------
# Menú principal
# -----------------------------
if __name__ == "__main__":
    ruta_md = "docs/procesos_academicos.md"
    base_faq = cargar_preguntas_desde_markdown(ruta_md)
    
    pregunta = input("❓ Ingresa tu pregunta sobre procesos académicos: ")
    buscar_respuesta(base_faq, pregunta)
