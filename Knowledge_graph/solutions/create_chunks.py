from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
import logging
import os
from dotenv import load_dotenv  # Import thư viện dotenv

# Load biến môi trường từ file .env
load_dotenv()

# Cấu hình logging
logging.basicConfig(format="%(asctime)s - %(message)s", level="INFO")

# Định nghĩa lớp CreateChunksofDocument
class CreateChunksofDocument:
    def __init__(self, pages: list[Document]):
        self.pages = pages

    def split_file_into_chunks(self, token_chunk_size, chunk_overlap):
        """
        Split a list of documents(file pages) into chunks of fixed size.

        Args:
            token_chunk_size: Size of each chunk in tokens.
            chunk_overlap: Overlap size between consecutive chunks.

        Returns:
            A list of chunks each of which is a langchain Document.
        """
        logging.info("Split file into smaller chunks")
        from langchain.text_splitter import TokenTextSplitter

        text_splitter = TokenTextSplitter(
            chunk_size=token_chunk_size,
            chunk_overlap=chunk_overlap
        )
        MAX_TOKEN_CHUNK_SIZE = int(os.getenv('MAX_TOKEN_CHUNK_SIZE', 10000))
        chunk_to_be_created = int(MAX_TOKEN_CHUNK_SIZE / token_chunk_size)

        # Split based on metadata availability
        if 'page' in self.pages[0].metadata:
            chunks = []
            for i, document in enumerate(self.pages):
                page_number = i + 1
                if len(chunks) >= chunk_to_be_created:
                    break
                else:
                    for chunk in text_splitter.split_documents([document]):
                        chunks.append(Document(page_content=chunk.page_content, metadata={'page_number': page_number}))
        else:
            chunks = text_splitter.split_documents(self.pages)

        chunks = chunks[:chunk_to_be_created]
        return chunks

# Tải tài liệu từ file PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return pages

# Đọc file PDF
pdf_file_path = "../data/Apple stock during pandemic.pdf"  # Thay đổi đường dẫn file PDF
pages = load_pdf(pdf_file_path)

# Khởi tạo đối tượng CreateChunksofDocument
document_chunker = CreateChunksofDocument(pages=pages)

# Tách tài liệu thành các chunks
chunks = document_chunker.split_file_into_chunks(token_chunk_size=500, chunk_overlap=100)
# print(f"Total number of chunks: {len(chunks)}")

# Hiển thị các chunk
# for chunk in chunks:
#     print(f"Page {chunk.metadata['page_number']}: {chunk.page_content[:200]}...")  # In ra 200 ký tự đầu tiên của mỗi chunk
