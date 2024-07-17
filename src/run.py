from dotenv import load_dotenv
from intent import IntentService
from response import ResponseService
from data import DataService

load_dotenv()

# PDF 파일
pdf = "data/spiderman1.pdf"

data_service = DataService()

#redis에서 모든 데이터 삭제
data_service.drop_redis_data()

# PDF에서 데이터 읽어서 redis로 넣기
data = data_service.pdf_to_embeddings(pdf)

data_service.load_data_to_redis(data)

intent_service = IntentService()
response_service = ResponseService()

question = "스파이더맨의 감독은 누구야?"
intents = intent_service.get_intent(question)
facts = data_service.search_redis(intents)
answer = response_service.generate_response(facts, question)
print(answer)