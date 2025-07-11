from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from solutions.create_chunks import CreateChunksofDocument


load_dotenv()

pdf_file_path = "data/Apple stock during pandemic.pdf"
pages = PyPDFLoader(pdf_file_path).load_and_split()
chunks = CreateChunksofDocument(pages=pages).split_file_into_chunks(
    token_chunk_size=500,
    chunk_overlap=100
)

# Initialize LLM
llm = ChatOpenAI(
    openai_api_key=["OPENAI_API_KEY"], 
    model_name="gpt-3.5-turbo"
)


# # Define the schema for the graph document
# schema = { 
#     "nodes": [
#         {"type": "Person", "properties": node_properties},
#         {"type": "Company", "properties": node_properties},
#         {"type": "Industry", "properties": node_properties},
#         {"type": "Event", "properties": node_properties},
#         {"type": "Pandemic", "properties": node_properties},
#         {"type": "Epidemic", "properties": node_properties}
#     ],
#     "relationships": [
#         {"type": "AUTHOR_OF", "properties": relationship_properties},
#         {"type": "AFFECTS", "properties": relationship_properties},
#         {"type": "LOCATED_IN", "properties": relationship_properties},
#         {"type": "HAS", "properties": relationship_properties},
#         {"type": "PRODUCES", "properties": relationship_properties},
#         {"type": "COMPETES_WITH", "properties": relationship_properties}
#     ]
# }

# Define allowed node and relationship types
allowed_node = {"Person", "Company", "Industry", "Event", "Pandemic", "Epidemic"}
allowed_relationship = {
    "AUTHOR_OF",
    "AFFECTS",
    "LOCATED_IN",
    "HAS",
    "PRODUCES",
    "COMPETES_WITH"
}

node_properties = True
relationship_properties = True
# Define the graph transformer
doc_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_node=allowed_node,
    allowed_relationship=allowed_relationship,
    # node_properties=node_properties,
    # relationship_properties=relationship_properties,
    strict_mode=True,   #loại bỏ bất kì thông tin nào không tuân thủ
    # schema=schema
)

# Process chunks and extract graph
for chunk in chunks:
    try:
        # Convert chunk to graph document
        graph_docs = doc_transformer.convert_to_graph_documents([chunk])
        
        # Filter extracted nodes and relationships
        filtered_nodes = [
            node for node in graph_docs.nodes
            if node.type in allowed_node
        ]
        filtered_relationships = [
            rel for rel in graph_docs.relationships
            if rel.type in allowed_relationship
        ]

        # Debugging outputs
        print(f"Chunk Metadata: {chunk.metadata}")
        print(f"Filtered Nodes: {filtered_nodes}")
        print(f"Filtered Relationships: {filtered_relationships}")

    except Exception as e:
        print(f"Error processing chunk: {e}")
