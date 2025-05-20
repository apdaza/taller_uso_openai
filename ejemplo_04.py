import json
import pandas as pd
from openai import OpenAI
import numpy as np
import config

client = OpenAI(project=config.OPENAI_PROJECT, api_key=config.OPENAI_API_KEY)


def estimar_costos():
    prompt = input("Ingresa el prompt para evaluar el costo: ")
    response = client.responses.create(model="gpt-4o", input=prompt)
    # Guaardamos el output.json con el contenido de la respuesta
    with open("output.json", "w", encoding="utf-8") as f:
        f.write(response.to_json(indent=4))
        
    respuesta_ia = response.output_text
    print("-------------------------------------------------------------------------------")
    print(respuesta_ia)
    print("-------------------------------------------------------------------------------")
    print("\n📊 Tokens utilizados:")
    print(f"Input: {response.usage.input_tokens} | Output: {response.usage.output_tokens}")
    print("-------------------------------------------------------------------------------")
    tarifa_entrada = 0.15 / 1_000_000
    tarifa_salida = 0.60 / 1_000_000
    total = response.usage.input_tokens * tarifa_entrada + response.usage.output_tokens * tarifa_salida
    print(f"💰 Costo estimado: ${total:.6f} USD")

if __name__ == "__main__":
    estimar_costos()
