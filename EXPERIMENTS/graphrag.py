import os
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import Neo4jVector

load_dotenv()

# graph = Neo4jGraph(
#                     url=self.URI,
#                     username="neo4j",
#                     password=self.AUTH[1],
#                     enhanced_schema=True,
#                 )

# we load info about the database from the .env file @Luca
URI = os.getenv('NEO4J_URI')
AUTH = (os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))

llm = AzureChatOpenAI(
    openai_api_version="2024-10-21",
    deployment_name="contact-Yogaexperiment_gpt4omini",
    temperature=0,
)

embedder = AzureOpenAIEmbeddings(
    deployment=os.environ.get("AZURE_DEPLOYMENT_EMBEDDINGS"),
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    azure_endpoint="https://iitlines-swecentral1.openai.azure.com",
    model="text-embedding-ada-002",
    chunk_size=1024 #fix this one
)



vector_index = Neo4jVector.from_existing_graph(
    embedder,
    url=URI,
    username="neo4j",
    password=AUTH[1],
    index_name="person_index",
    node_label="Person",
    text_node_properties=["id", "trainingLevel"],
    embedding_node_property="embedding",
)

vector_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_index.as_retriever()
)
print(vector_qa.invoke(
    "tell me the sports practiced by liam patel",
))
