from openai import OpenAI
import config

client = OpenAI(project=config.OPENAI_PROJECT, api_key=config.OPENAI_API_KEY)


def asistente_redaccion():
    tema = input("Tema del ensayo académico: ")
    prompt = f"""
    Redacta un ensayo académico con estructura de introducción, desarrollo y conclusión sobre el tema: {tema}.
    El texto debe tener un tono formal, estar bien organizado y utilizar argumentos académicos válidos.
    """
    response = client.responses.create(
        model="gpt-4o",
        input=prompt,
        max_output_tokens=800
    )
    print("\n📄 Ensayo generado:\n")
    print(response.output_text)

if __name__ == "__main__":
    asistente_redaccion()
