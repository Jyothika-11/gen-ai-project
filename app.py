import os
from dotenv import load_dotenv



# Load environment variables from .env file
load_dotenv()

# Access them using os.getenv
api_key = os.getenv("OPENAI_API_KEY")
db_user = os.getenv("DB_USER")
db_pass = os.getenv("DB_PASS")

print("API Key:", api_key)
print("Database User:", db_user)
print("Database Password:", db_pass)


#from dotenv import load_dotenv

#load_dotenv()  # loads variables from a .env file into environment

from langchain_community.document_loaders import PyPDFLoader
#from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI

from langchain.chains import RetrievalQA

print("API key loaded:",bool(os.getenv("OPENAI_API_KEY")))
#load environment variables
#load_dotenv()

def load_pdf(pdf_path):
    loader=PyPDFLoader(pdf_path)
    documents=loader.load()
    return documents

def split_documents(documents):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

def create_vector_store(docs):
    embeddings=OpenAIEmbeddings()
    vectorstore=FAISS.from_documents(docs,embeddings)
    return vectorstore

def create_qa_chain(vectorstore):
    llm=OpenAI(temperature=0)
    qa_chain=RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    return qa_chain
def main():
    print("GenAI Document Q&A assistant")
    print("-" * 40)

    pdf_path="data/Generative AI.pdf"

    if not os.path.exists(pdf_path):
        print("pdf file not found.add file")
        return
    
    print("loading document...")
    documents=load_pdf(pdf_path)

    print("splitting text...")
    docs=split_documents(documents)

    print("creating vector store..")
    vectorstore=create_vector_store(docs)

    print("system ready! ask questions (type 'exit' to quit)\n")

    qa_chain=create_qa_chain(vectorstore)

    while True:
        query=input("Question: ")
        if query.lower()=="exit":
            print("Goodbye")
            break
        answer=qa_chain.run(query)
        print(f"\n answe:\n{answer}\n")

if __name__ == "__main__":
    main()
    