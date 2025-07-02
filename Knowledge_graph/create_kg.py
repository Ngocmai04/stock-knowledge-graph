import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

DOCS_PATH = "data/*.pdf"

# Khởi tạo mô hình LLM và công cụ embedding
llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name="gpt-3.5-turbo"
)

embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model="text-embedding-ada-002"
)

# Khởi tạo đồ thị Neo4j
graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

# Khởi tạo transformer để chuyển đổi văn bản thành graph document
doc_transformer = LLMGraphTransformer(llm=llm)

# Hàm tìm node tương tự trong đồ thị
def find_similar_node(embedding, threshold=0.8):
    """
    Tìm node có embedding tương tự trong đồ thị Neo4j.
    """
    # Truy vấn để lấy tất cả node và embedding
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

# Load và chia nhỏ tài liệu
loader = DirectoryLoader(DOCS_PATH, glob="data/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n\n", 
    chunk_size=1500,   
    chunk_overlap=200, 
)
chunks = text_splitter.split_documents(docs)

# Xử lý từng đoạn (chunk) và tạo node trong đồ thị
for chunk in chunks:
    filename = os.path.basename(chunk.metadata["source"])  # Lấy tên file
    chunk_id = f"{filename}.{chunk.metadata['page']}"      # Tạo ID cho mỗi chunk
    print("Đang xử lý -", chunk_id)

    # Tạo embedding cho đoạn văn bản
    chunk_embedding = embedding_provider.embed_query(chunk.page_content)

    # Thêm node Document và Chunk vào đồ thị
    properties = {
        "filename": filename,
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "embedding": chunk_embedding
    }

    graph.query("""
        MERGE (d:Document {id: $filename})  # Tạo hoặc cập nhật node Document
        MERGE (c:Chunk {id: $chunk_id})    # Tạo hoặc cập nhật node Chunk
        SET c.text = $text                 # Cập nhật nội dung cho Chunk
        MERGE (d)<-[:PART_OF]-(c)          # Tạo quan hệ giữa Document và Chunk
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)  # Gắn embedding cho Chunk
    """, properties)

    # Trích xuất các node và quan hệ từ văn bản
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])

    for graph_doc in graph_docs:
        chunk_node = Node(id=chunk_id, type="Chunk")  # Tạo node Chunk tạm thời

        for node in graph_doc.nodes:
            # Tìm node tương tự
            similar_node_id = find_similar_node(node.embedding, threshold=0.8)

            if similar_node_id:
                # Nếu tìm thấy node tương tự, tạo quan hệ với node đó
                graph.query("""
                    MATCH (c:Chunk {id: $chunk_id}), (e:Entity {id: $similar_node_id})
                    MERGE (c)-[:HAS_ENTITY]->(e)
                """, {"chunk_id": chunk_id, "similar_node_id": similar_node_id})
            else:
                # Nếu không tìm thấy node tương tự, tạo node mới
                graph.query("""
                    CREATE (e:Entity {id: $entity_id, name: $entity_name, embedding: $embedding})
                    MERGE (c:Chunk {id: $chunk_id})-[:HAS_ENTITY]->(e)
                """, {
                    "entity_id": node.id,
                    "entity_name": node.properties.get("name", ""),
                    "embedding": node.embedding,
                    "chunk_id": chunk_id,
                })

# Tạo vector index để tìm kiếm nhanh trong đồ thị
graph.query("""
    CREATE VECTOR INDEX `chunkVector`
    IF NOT EXISTS
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }};
""")
