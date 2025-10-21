import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

PDF_FILE_PATH = "document.pdf"

if not os.path.exists(PDF_FILE_PATH):
    print(f"ERROR: The file '{PDF_FILE_PATH}' was not found.")
    print("Please make sure you have a PDF named 'document.pdf' in the same folder.")
    exit()

print(f"Loading and processing {PDF_FILE_PATH}...")

#Load the PDF document
try:
    loader = PyPDFLoader(PDF_FILE_PATH)
    data = loader.load()
except Exception as e:
    print(f"Error loading PDF: {e}")
    exit()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(data)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_model)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 most relevant chunks

print("PDF processed successfully. Ready to analyze.")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#prompt template
template = """
You are an expert PDF analysis assistant. Your only source of truth is the provided context.
Answer the question based SOLELY on the following context.
If the answer is not found in the context, clearly state that the information is not available in the document.
Do not use any external knowledge.

Context:
{context}

Question: {question}

Answer:
"""
RAG_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": RAG_PROMPT}
)

print("\n--- PDF Analysis Agent ---")
print("Ask a question about the 'document.pdf' content.")
print("Type 'exit' to quit.")

while True:
    try:
        query = input("\nYour Question: ")
        if query.lower() == 'exit':
            break

        if not query:
            continue

        response = qa_chain.invoke({"query": query})
        
        print("-" * 50)
        print(response.get("result"))
        print("-" * 50)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        break

print("\nAnalysis session ended. Goodbye!")
