import json
import pandas as pd
from openai import OpenAI
import numpy as np
import config

client = OpenAI(project=config.OPENAI_PROJECT, api_key=config.OPENAI_API_KEY)


def clasificar_opinion():
    opinion = input("Ingresa una opinión estudiantil sobre el curso: ")
    instrucciones = f"""
    [ROL] Eres un experto en clasificación de opiniones estudiantiles.
    [OBJETIVO] Clasifica la siguiente opinión según la satisfacción y posibles áreas de mejora.
    [OPINION] {opinion}
    [SALIDA] JSON con campos: {{'clasificacion':'...', 'area_mejora':'...'}}
    """
    response = client.responses.create(
        model="gpt-4o",
        input=instrucciones,
        text={"format": {"type": "json_object"}},
        max_output_tokens=200
    )
    resultado = json.loads(response.output_text)
    print("\n🎯 Resultado:")
    print(json.dumps(resultado, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    clasificar_opinion()
