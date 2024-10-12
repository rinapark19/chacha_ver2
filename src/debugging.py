import openai
from langchain.llms import OpenAI

def moderate_content(text):
    response = openai.moderations.create(input=text)

    if response.results[0].flagged:
        return "부적절한 콘텐츠가 감지되었습니다. 문장을 다시 입력해 주세요."
    return text

moderated_text = moderate_content("You are idiot")
print(moderated_text)