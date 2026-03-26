import os
from openai import AzureOpenAI

endpoint = "https://comsi-md4b9qgt-eastus2.cognitiveservices.azure.com/"
model_name = "gpt-5.4-nano"
deployment = "gpt-5.4-nano"

subscription_key = "1lpVqQuGLPAobeHfyIuMmX8yQDh8Iq7tPRFxEDIyPPr5fQRJsx1EJQQJ99BGACHYHv6XJ3w3AAAAACOGRB4P"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_completion_tokens=16384,
    model=deployment
)

print(response.choices[0].message.content)