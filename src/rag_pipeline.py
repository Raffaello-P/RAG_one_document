from src.ollama_client import OllamaClient

class RAGpipeline:
    def __init__(self, llm_config):
        self.llm_client = OllamaClient(llm_config)
        

    def query(self, full_prompt):
        #context = ""
        #context_text = "\n".join([doc['content'] for doc in context])
        return self.llm_client.query(full_prompt)