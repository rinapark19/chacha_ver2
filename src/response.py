import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ResponseService():
    def __init__(self):
        pass
    
    def generate_response(self, facts, user_question):
        response = client.chat.completions.create(model="gpt=3.5-turbo",
                                                  messages = [
            {"role": "user",
             "content": "Based on the FACTS, give an answer to the question." + f"QUESTION: {user_question}"}
        ])

        return (response.choices[0].message.content)
        