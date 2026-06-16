# import aisuite as ai
from dotenv import load_dotenv
import os

# load_dotenv()


# client = ai.Client()

# models = ["openai:gpt-5.2", "anthropic:claude-3-5-sonnet-20241022", "groq:llama-3.2-3b-preview"]

# messages = [
#     {"role": "system", "content": "Respond in Pirate English."},
#     {"role": "user", "content": "Tell me a joke."},
# ]

# for model in models:
#     response = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=0.75
#     )
#     print(response.choices[0].message.content)
    
import aisuite as ai
from aisuite.provider import ProviderFactory

load_dotenv()

# Get list of supported providers
providers = ProviderFactory.get_supported_providers()
print("Supported providers:", providers)

client = ai.Client()

# model="google:gemini-1.5-pro-001"

# messages = [
#     {"role": "system", "content": "Respond in Pirate English."},
#     {"role": "user", "content": "Tell me a joke."},
# ]

# response = client.chat.completions.create(
#     model=model,
#     messages=messages,
# )

# print(response.choices[0].message.content)

provider = "deepseek"
model_id = "deepseek-reasoner"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What’s the weather like in San Francisco?"},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)