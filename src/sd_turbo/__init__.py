import torch
from diffusers import AutoPipelineForText2Image

print("Cargando modelo....")

modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype = torch.float32
    #El floatXX puede ser 32 o 16
    # Me indica la cantidad de decimales de precision
    # con el que el model creara la imagen
    # si mi pc me genera una imagen negra, cambio a 16
)

modelo = modelo.to("cuda")
# Si tengo un pc xqria usar cpu
# Si tego nvidia poner cuda
# Si tengo mac con procesadores MX poner mps
# Si tengo grafica intel IRIS xpu

prompt = input("Escribe el prompt de la imagen:")
negative_prompt = "worst quality, low quality, normal quality, low-res, blurry, jpeg artifacts, text, watermark, signature, username, error, bad anatomy, bad hands, extra fingers, fewer digits, cropped, deformed face, disfigured, ugly"

print("Generando la imagen...")

imagen = modelo(
    prompt = prompt,
    negative_prompt = negative_prompt,
    num_inference_steps = 10, #Numero de veces que creará y corregiá la imagen
    guidance_scale = 2.0, #Que tan fiel será al prompt
).images[0]

imagen.save("image.png")
print("Imagen guardada!")
