import os
from dotenv import load_dotenv
from openai import OpenAI

# 여기에 나중에 말투 프롬프팅

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class IntentService():
    def __init__(self):
        pass

    def get_intent(self, user_question: str):
        # OpenAI API 호출
        response = client.completions.create(model="gpt-3.5-turbo",messages = [
            {"role": "user", 
             "content": f"Extract the keywords from the following question: {user_question}" + "Do not answer anything else, only the keywords"}
        ])
        

        return (response.choices[0].message.content)