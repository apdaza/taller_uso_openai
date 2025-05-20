from openai import OpenAI
import config


client = OpenAI(project=config.OPENAI_PROJECT, api_key=config.OPENAI_API_KEY)


response = client.responses.create(
    model="gpt-4o-mini", #Alias
    input="Cuentame un chiste"
)


# Guaardamos el output.json con el contenido de la respuesta
with open("output.json", "w", encoding="utf-8") as f:
    f.write(response.to_json(indent=4))
    
respuesta_ia = response.output_text
    
print(respuesta_ia)
print("Uso de entrada: ", response.usage.input_tokens)
print("Uso de salida: ",response.usage.output_tokens)

# Calcular consumo
tarifa_entrada = (0.15/1000000)
tarifa_salida = (0.6/1000000)

consumo_entrada = response.usage.input_tokens * tarifa_entrada
consumo_salida = response.usage.output_tokens * tarifa_salida
consumo_total = consumo_entrada + consumo_salida
print("Consumo total: ", consumo_total)