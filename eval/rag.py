from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
import os

text = '''Our loaded document is over 42k characters long. This is too long to fit in the context window of many models. Even for those models that could fit the full post in their context window, models can struggle to find information in very long inputs.

To handle this we’ll split the Document into chunks for embedding and vector storage. This should help us retrieve only the most relevant bits of the blog post at run time.

In this case we’ll split our documents into chunks of 1000 characters with 200 characters of overlap between chunks. The overlap helps mitigate the possibility of separating a statement from important context related to it. We use the RecursiveCharacterTextSplitter, which will recursively split the document using common separators like new lines until each chunk is the appropriate size. This is the recommended text splitter for generic text use cases.

We set add_start_index=True so that the character index at which each split Document starts within the initial Document is preserved as metadata attribute “start_index”.'''

def rag(text, query, length=4096):
    documents = Document(page_content=text)
    top_k = length//100
    
    text_splitter =RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "?", "."],
        chunk_size=100,
        chunk_overlap=0,
    )
    texts = text_splitter.split_documents([documents])
    vectorstore = Chroma.from_documents(documents=texts, embedding=HuggingFaceEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    retrieved_docs = retriever.invoke(query)

    sim_doc = ''
    for i in range(min(top_k, len(retrieved_docs))):
        sim_doc += retrieved_docs[i].page_content
        
    return sim_doc

a = rag(text, 'who helps us?')
print(a)
