import os
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import redis
import redis.exceptions

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_NAME = "embeddings-index"
PREFIX = "doc"
DISTANCE_METRIC = "COSINE" # IP, L2

REDIS_HOST = "localhost"
REDIS_PORT = 6379

class DataService():
    def __init__(self):
        # 레디스에 연결
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True
        )

        try:
            self.redis_client.ping()
            print(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except redis.exceptions.ConnectionError as e:
            print(f"Failed to connect to Redis: {e}")
    
    def drop_redis_data(self, index_name: str = INDEX_NAME):
        # 인덱스 제거
        try:
            self.redis_client.ft(index_name).dropindex()
            print("Index dropped")
        except:
            print("Index does not exist")
    
    def load_data_to_redis(self, embeddings):
        vector_dim = len(embeddings[0]['vector']) # 벡터 길이
        vector_number = len(embeddings) # 초기 벡터 개수

        # RedisSearch fields 정의
        text = TextField(name="text")
        text_embedding = VectorField("vector",
                                     "FLAT", {
                                         "TYPE": "FLOAT32",
                                         "DIM": vector_dim,
                                         "DISTANCE_METRIC": DISTANCE_METRIC,
                                         "INITIAL_CAP": vector_number,
                                     })
        fields = [text, text_embedding]

        # 인덱스 있는지 확인
        try:
            self.redis_client.ft(INDEX_NAME).info()
            print("Index already exists")
        
        except:
            # 인덱스 생성
            self.redis_client.ft(INDEX_NAME).create_index(
                fields=fields,
                definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
            )
        
        for embedding in embeddings:
            key = f"{PREFIX}:{str(embedding['id'])}"
            embedding["vector"] = np.array(embedding["vector"], dtyp=np.float32).tobytes()
            self.redis_client.hset(key, mapping=embedding)
        print(f"Loaded {self.redis_client.info()['db0']['keys']} documents in Redis search index with name: {INDEX_NAME}")

    def pdf_to_embeddings(self, pdf_path: str, chunk_length: int=1000):
        # PDF에서 데이터 읽어서 chunk로 분리
        reader = PdfReader(pdf_path)
        chunks = []

        for page in reader.pages: # tqdm 추가
            text_page = page.extract_text()
            chunks.extend([text_page[i:i+chunk_length].replace("\n", "") for i in range(0, len(text_page), chunk_length)])


        # embedding 생성
        response = client.embeddings.create(model="text-embedding-ada-002", input=chunks)
        return [{"id": value.index, 'vector': value.embedding, 'text': chunks[value.index]} for value in response.data]
    
    def search_redis(self, user_query: str, index_name: str = INDEX_NAME, vector_field: str = "vector", return_fields: list = ["text", "vector_score"], hybrid_fields="*",
                     k: int = 5, print_results: bool = False,):
        
        # 사용자 쿼리로 임베딩 벡터 만들기
        embedded_query = client.embeddings.create(input=user_query, model="text-embedding-ada-002").data[0].embedding

        # 쿼리 준비
        base_query = f"{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]"
        query = (Query(base_query).return_fields(return_fields).sort_by("vector_score").paging(0, k).dialect(2))
        params_dict = {"vector": np.array(embedded_query).astype(dtype=np.float32).tobytes()}

        # 벡터 서치 실행
        results = self.redis_client.ft(index_name).search(query, params_dict)
        if print_results:
            for i, doc in enumerate(results.docs):
                score = 1 - float(doc.vector_score)
                print(f"{i}. {doc.text} (Score: {round(score, 3) })")
        return [doc['text'] for doc in results.docs]