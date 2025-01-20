import os
import yaml
from src.rag_pipeline import RAGpipeline
from src.chunkSplit import pdf_to_chunks, pdf_to_chunks_with_langchain

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml") #fornire path di configurazione
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    #configurazione percorsi e template
    documents_dir =  os.path.join(os.path.dirname(__file__), config["paths"]["documents_dir"])
    print(f"document dir: {documents_dir}")
    pdf_path = os.path.join(documents_dir, "document.pdf") # inserire nome del documento
    print(f"pdf_path: {pdf_path}")
    prompt_template = config["prompt"]

    # creazione chunks
    max_tokens=config["llm"]["max_tokens"]
    #chunks = pdf_to_chunks(pdf_path)
    chunks= pdf_to_chunks_with_langchain(pdf_path, max_tokens)
    print(f"Diviso in {len(chunks)} chunk.")

    #configurazione pipline del sistema RAG
    pipeline = RAGpipeline(llm_config=config["llm"]) # Passo la configurazione llm del file yaml

    print("Benvenuto! Digita una domanda per il modello (digita 'exit' per uscire).")
    while True:
        user_query = input(">> ")
        if user_query.lower() == "exit":
            print("Arrivederci!")
            break

        full_prompt = prompt_template.format(context=chunks, query=user_query)
        
        response = pipeline.query(full_prompt)
        print(f"Risposta: {response["response"]}")

if __name__ == "__main__":
    main()
