import os
import asyncio
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import logger, set_verbose_debug
from dotenv import load_dotenv
import time
import pathlib
from docling.document_converter import DocumentConverter

load_dotenv()

WORKING_DIR = "./bank_dir"

def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_demo.log"))

    print(f"\nLightRAG demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,   
        chunk_token_size=200,
        chunk_overlap_token_size=40  
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

def pdf_to_txt(pdf_path: str, out_txt_path: str):
    pdf_path = pathlib.Path(pdf_path)
    out_txt_path = pathlib.Path(out_txt_path)

    # Initialize converter
    converter = DocumentConverter()
    # Convert the document
    result = converter.convert(str(pdf_path))
    
    out_txt_file = out_txt_path.with_suffix(".txt")
    
    text = result.document.export_to_text()
    with out_txt_file.open("w", encoding="utf-8") as fp:
        fp.write(text)

async def main():
    # Check if OPENAI_API_KEY environment variable exists
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable is not set. Please set this variable before running the program."
        )
        print("You can set the environment variable by running:")
        print("  export OPENAI_API_KEY='your-openai-api-key'")
        return  # Exit the async function

    try:
        # Clear old data files
        files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
        ]

        # for file in files_to_delete:
        #     file_path = os.path.join(WORKING_DIR, file)
        #     if os.path.exists(file_path):
        #         os.remove(file_path)
        #         print(f"Deleting old file:: {file_path}")

        # Initialize RAG instance
        rag = await initialize_rag()

        # Test embedding function
        # test_text = ["This is a test string for embedding."]
        # embedding = await rag.embedding_func(test_text)
        # embedding_dim = embedding.shape[1]
        # print("\n=======================")
        # print("Test embedding function")
        # print("========================")
        # print(f"Test dict: {test_text}")
        # print(f"Detected embedding dimension: {embedding_dim}\n\n")

        # doc_file_path = 'bank.pdf'

        # pdf_to_txt(doc_file_path, 'bank_output')
        
        # with open("./bank_output.txt", "r", encoding="utf-8") as f:
        #     await rag.ainsert(f.read())
        
        # text_content_bytes = textract.process(doc_file_path)
        # text_content = text_content_bytes.decode('utf-8')

        # with open('bank_output.txt', 'w', encoding='utf-8') as f:
        #     f.write(text_content)
        
        # await rag.ainsert(text_content)
        
        print("\n=====================")
        print("Query mode: NAIVE")
        print("=====================")
        
        start_time = time.perf_counter()  # Start timing
        
        print(
            await rag.aquery(
                "Find all borrowers who are jointly linked to more than one partner RE via the same CLA, and check which escrow banks serve as intermediaries.", param=QueryParam(mode="naive")
            )
        )
        
        end_time = time.perf_counter()  # End timing

        elapsed_time = end_time - start_time
        print(f"Query execution time-1: {elapsed_time * 1000:.2f} ms")
        print("----------------------------------------------------end of 1-answer----------------------------------------------------\n")
        
        
        # # Perform local search
        # print("\n=====================")
        # print("Query mode: naive-2")
        # print("=====================")
        
        # start_time = time.perf_counter()  # Start timing
        
        # print(
        #     await rag.aquery(
        #         "What is the overall message about social responsibility in this story?", param=QueryParam(mode="naive", top_k=80, chunk_top_k=15)
        #     )
        # )
        
        # end_time = time.perf_counter()  # End timing
        
        # elapsed_time = end_time - start_time
        # print(f"Query execution time-2: {elapsed_time * 1000:.2f} ms")
        # print("----------------------------------------------------end of 2-answer----------------------------------------------------\n")


        print("\n=====================")
        print("Query mode: LOCAL")
        print("=====================")
        
        start_time = time.perf_counter()  # Start timing
        
        print(
            await rag.aquery(
                "Find all borrowers who are jointly linked to more than one partner RE via the same CLA, and check which escrow banks serve as intermediaries.", param=QueryParam(mode="local")
            )
        )
        
        end_time = time.perf_counter()  # End timing

        elapsed_time = end_time - start_time
        print(f"Query execution time-1: {elapsed_time * 1000:.2f} ms")
        print("----------------------------------------------------end of 1-answer----------------------------------------------------\n")
        
        
        # # Perform local search
        # print("\n=====================")
        # print("Query mode: naive-2")
        # print("=====================")
        
        # start_time = time.perf_counter()  # Start timing
        
        # print(
        #     await rag.aquery(
        #         "What does Tiny Tim say at the end of the story?", param=QueryParam(mode="naive", top_k=80, chunk_top_k=15)
        #     )
        # )
        
        # end_time = time.perf_counter()  # End timing
        
        # elapsed_time = end_time - start_time
        # print(f"Query execution time-2: {elapsed_time * 1000:.2f} ms")
        # print("----------------------------------------------------end of 2-answer----------------------------------------------------\n") 
        
        print("\n=====================")
        print("Query mode: GLOBAL")
        print("=====================")
        
        start_time = time.perf_counter()  # Start timing
        
        print(
            await rag.aquery(
                "Find all borrowers who are jointly linked to more than one partner RE via the same CLA, and check which escrow banks serve as intermediaries.", param=QueryParam(mode="global")
            )
        )
        
        end_time = time.perf_counter()  # End timing

        elapsed_time = end_time - start_time
        print(f"Query execution time-1: {elapsed_time * 1000:.2f} ms")
        print("----------------------------------------------------end of 1-answer----------------------------------------------------\n")
        
        
        # # Perform local search
        # print("\n=====================")
        # print("Query mode: naive-2")
        # print("=====================")
        
        # start_time = time.perf_counter()  # Start timing
        
        # print(
        #     await rag.aquery(
        #         "Who are the three spirits that visit Scrooge?", param=QueryParam(mode="naive", top_k=80, chunk_top_k=15)
        #     )
        # )
        
        # end_time = time.perf_counter()  # End timing
        
        # elapsed_time = end_time - start_time
        # print(f"Query execution time-2: {elapsed_time * 1000:.2f} ms")
        # print("----------------------------------------------------end of 2-answer----------------------------------------------------\n")
        
        
        print("\n=====================")
        print("Query mode: HYBRID")
        print("=====================")
        
        start_time = time.perf_counter()  # Start timing
        
        print(
            await rag.aquery(
                "Find all borrowers who are jointly linked to more than one partner RE via the same CLA, and check which escrow banks serve as intermediaries.", param=QueryParam(mode="hybrid")
            )
        )
        
        end_time = time.perf_counter()  # End timing

        elapsed_time = end_time - start_time
        print(f"Query execution time-1: {elapsed_time * 1000:.2f} ms")
        print("----------------------------------------------------end of 1-answer----------------------------------------------------\n")
        
        
        # # Perform local search
        # print("\n=====================")
        # print("Query mode: naive-2")
        # print("=====================")
        
        # start_time = time.perf_counter()  # Start timing
        
        # print(
        #     await rag.aquery(
        #         "How is Tiny Tim related to other characters in the story?", param=QueryParam(mode="naive", top_k=80, chunk_top_k=15)
        #     )
        # )
        
        # end_time = time.perf_counter()  # End timing
        
        # elapsed_time = end_time - start_time
        # print(f"Query execution time-2: {elapsed_time * 1000:.2f} ms")
        # print("----------------------------------------------------end of 2-answer----------------------------------------------------\n")
        
        print("\n=====================")
        print("Query mode: MIX")
        print("=====================")
        
        start_time = time.perf_counter()  # Start timing
        
        print(
            await rag.aquery(
                "Find all borrowers who are jointly linked to more than one partner RE via the same CLA, and check which escrow banks serve as intermediaries.", param=QueryParam(mode="mix")
            )
        )
        
        end_time = time.perf_counter()  # End timing

        elapsed_time = end_time - start_time
        print(f"Query execution time-1: {elapsed_time * 1000:.2f} ms")
        print("----------------------------------------------------end of 1-answer----------------------------------------------------\n")
        
        
        # # Perform local search
        # print("\n=====================")
        # print("Query mode: naive-2")
        # print("=====================")
        
        # start_time = time.perf_counter()  # Start timing
        
        # print(
        #     await rag.aquery(
        #         "What is the relationship between Scrooge and Jacob Marley?", param=QueryParam(mode="naive", top_k=80, chunk_top_k=15)
        #     )
        # )
        
        # end_time = time.perf_counter()  # End timing
        
        # elapsed_time = end_time - start_time
        # print(f"Query execution time-2: {elapsed_time * 1000:.2f} ms")
        # print("----------------------------------------------------end of 2-answer----------------------------------------------------\n")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
