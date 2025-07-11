import logging
from typing import List
from pydantic.v1 import BaseModel, Field
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Tải các biến môi trường từ file .env
load_dotenv()

# Hàm khởi tạo LLM (Mô hình ngôn ngữ lớn)
def get_llm(model_name: str):
    """
    Initializes a ChatOpenAI model instance.
    :param model_name: The name of the OpenAI model to use (e.g., "gpt-4").
    :return: An instance of ChatOpenAI.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("OPENAI_API_KEY is missing in .env file.")
    try:
        return ChatOpenAI(
            openai_api_key=api_key,
            model=model_name,
        )
    except Exception as e:
        raise Exception(f"Error initializing LLM: {e}")

# Định nghĩa schema Knowledge Graph
class Schema(BaseModel):
    """Knowledge Graph Schema."""
    triplets: List[str] = Field(
        description="List of node labels and relationship types in a graph schema in <NodeType1>-<RELATIONSHIP_TYPE>-><NodeType2> format"
    )

# Mẫu Prompt để trích xuất schema
PROMPT_TEMPLATE = """
You are an expert in schema extraction, especially in identifying node and relationship types from example texts.
Analyze the following text and extract only the types of entities (node types) and their relationship types.
Do not return specific instances or attributes — only abstract schema information.
Return the result in the following format:
{{"triplets": ["<NodeType1>-<RELATIONSHIP_TYPE>-><NodeType2>"]}}
For example, if the text says “John works at Microsoft”, the output should be:
{{"triplets": ["Person-WORKS_AT->Company"]}}
"""

# Hàm trích xuất nội dung từ PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.
    :param pdf_path: Path to the PDF file.
    :return: Combined text from all pages of the PDF.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            raise ValueError("No extractable text found in the PDF.")
        return text
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
        raise Exception(f"Could not extract text from PDF: {e}")

# Hàm trích xuất schema từ văn bản
def schema_extraction_from_text(input_text: str, llm: ChatOpenAI) -> List[str]:
    """
    Extract schema from input text using LLM.
    :param input_text: Text to process.
    :param llm: Initialized instance of ChatOpenAI.
    :return: Extracted schema as a list of triplets.
    """
    try:
        # Tạo prompt cho LLM
        prompt = ChatPromptTemplate.from_messages(
            [("system", PROMPT_TEMPLATE), ("user", "{text}")]
        )

        # Gửi prompt và xử lý kết quả
        runnable = prompt | llm.with_structured_output(
            schema=Schema,
            method="function_calling",
            include_raw=False,
        )

        raw_schema = runnable.invoke({"text": input_text})
        if raw_schema and raw_schema.triplets:
            return raw_schema.triplets
        else:
            raise Exception("Unable to extract schema from text.")
    except Exception as e:
        logging.error(f"Schema extraction error: {e}")
        raise Exception(str(e))

# Hàm chính để trích xuất schema từ PDF
def extract_schema_from_pdf(pdf_path: str, model_name: str) -> List[str]:
    """
    Main function to extract schema from a PDF file.
    :param pdf_path: Path to the PDF file.
    :param model_name: Name of the LLM model.
    :return: Extracted schema as a list of triplets.
    """
    logging.info(f"Extracting text from PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)

    logging.info("Initializing LLM...")
    llm = get_llm(model_name)

    logging.info("Extracting schema from text...")
    schema = schema_extraction_from_text(text, llm)
    return schema

if __name__ == "__main__":
    import argparse

    # Cấu hình parser cho dòng lệnh
    parser = argparse.ArgumentParser(description="Extract schema from a PDF file.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4"),
                        help="Name of the LLM model to use (default: model in .env)")

    args = parser.parse_args()

    # Thực thi trích xuất schema
    try:
        logging.basicConfig(level=logging.INFO)
        # pdf_path = "data/Apple stock during pandemic.pdf"
        extracted_schema = extract_schema_from_pdf(args.pdf_path, args.model)
        print("Extracted Schema:")
        for triplet in extracted_schema:
            print(f"- {triplet}")
    except Exception as e:
        print(f"Error: {e}")
