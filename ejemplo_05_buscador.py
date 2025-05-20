# buscador_faq_embed.py
import pandas as pd
import numpy as np
from openai import OpenAI
import config

client = OpenAI(project=config.OPENAI_PROJECT, api_key=config.OPENAI_API_KEY)

def get_embedding(texto):
    texto = texto.replace("\n", " ")
    return client.embeddings.create(input=[texto], model="text-embedding-ada-002").data[0].embedding

def similitud_coseno(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def responder_pregunta(pregunta_usuario):
    df = pd.read_csv("docs/faq_embed.csv")
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)

    vector_usuario = get_embedding(pregunta_usuario)
    df['similitud'] = df['embedding'].apply(lambda x: similitud_coseno(x, vector_usuario))
    mejor = df.sort_values(by='similitud', ascending=False).iloc[0]

    print(f"\n❓ Pregunta más parecida:\n{mejor['pregunta']}")
    print(f"💡 Respuesta sugerida:\n{mejor['respuesta']}")
    print(f"🔢 Similitud: {mejor['similitud']:.4f}")

if __name__ == "__main__":
    pregunta = input("Haz tu pregunta sobre procesos académicos: ")
    responder_pregunta(pregunta)
