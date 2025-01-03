import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perplexity_api_client import Perplexity
from dotenv import load_dotenv

load_dotenv()

pplx_ai = Perplexity(
    api_key=os.environ.get('PPLX_API_KEY'),
    model="llama-3.1-sonar-large-128k-online",
    system_role="You are a helpful assistant.",
)

input_message = "Enter a message. Enter 'exit' to quit: "

message = input(input_message)
while message != "exit":
    print(pplx_ai.chat(message))
    message = input(input_message)

pplx_ai.close()
print("Goodbye!")