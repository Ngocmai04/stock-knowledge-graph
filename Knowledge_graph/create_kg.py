import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node
from sklearn.metrics.pairwise import cosine_similarity

from solutions.create_chunks import CreateChunksofDocument

# Load biến môi trường
load_dotenv()

# === 1. Đọc và chia nhỏ tài liệu PDF ===
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return pages

pdf_file_path = "../data/Apple stock during pandemic.pdf"
if not os.path.exists(pdf_file_path):
    raise FileNotFoundError(f"Không tìm thấy file PDF tại: {pdf_file_path}")

pages = load_pdf(pdf_file_path)
if not pages:
    raise ValueError("Không load được trang nào từ PDF!")

document_chunker = CreateChunksofDocument(pages=pages)
chunks = document_chunker.split_file_into_chunks(token_chunk_size=500, chunk_overlap=100)
print(f"✅ Đã chia thành {len(chunks)} đoạn văn bản.")

# === 2. Khởi tạo mô hình ===
llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name="gpt-3.5-turbo"
)

embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model="text-embedding-ada-002"
)

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

doc_transformer = LLMGraphTransformer(llm=llm)

# === 3. Tìm node tương tự theo embedding ===
def find_similar_node(embedding, threshold=0.8):
    existing_nodes = graph.query("""
        MATCH (n:Entity)
        RETURN n.id AS id, n.embedding AS embedding
    """).data()

    for node in existing_nodes:
        if node["embedding"]:
            similarity = cosine_similarity([embedding], [node["embedding"]])[0][0]
            if similarity > threshold:
                return node["id"]
    return None

# === 4. Nhập dữ liệu vào Neo4j ===
for chunk in chunks:
    filename = os.path.basename(chunk.metadata["source"])
    chunk_id = f"{filename}.{chunk.metadata['page']}"
    print(f"📄 Đang xử lý: {chunk_id}")

    chunk_embedding = embedding_provider.embed_query(chunk.page_content)

    # Tạo node Document và Chunk
    properties = {
        "filename": filename,
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "embedding": chunk_embedding
    }

    graph.query("""
        MERGE (d:Document {id: $filename})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text
        MERGE (d)<-[:PART_OF]-(c)
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
    """, properties)

    # Trích xuất thực thể từ đoạn chunk
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])

    for graph_doc in graph_docs:
        for node in graph_doc.nodes:
            similar_node_id = find_similar_node(node.embedding, threshold=0.8)

            if similar_node_id:
                graph.query("""
                    MATCH (c:Chunk {id: $chunk_id}), (e:Entity {id: $similar_node_id})
                    MERGE (c)-[:HAS_ENTITY]->(e)
                """, {"chunk_id": chunk_id, "similar_node_id": similar_node_id})
            else:
                graph.query("""
                    CREATE (e:Entity {id: $entity_id, name: $entity_name, embedding: $embedding})
                    MERGE (c:Chunk {id: $chunk_id})-[:HAS_ENTITY]->(e)
                """, {
                    "entity_id": node.id,
                    "entity_name": node.properties.get("name", ""),
                    "embedding": node.embedding,
                    "chunk_id": chunk_id
                })

# === 5. Tạo chỉ mục vector để truy vấn nhanh ===
graph.query("""
    CREATE VECTOR INDEX `chunkVector`
    IF NOT EXISTS
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }};
""")

print("✅ Đã hoàn tất việc xây dựng Knowledge Graph!")
