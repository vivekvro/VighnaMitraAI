from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

embedding = HuggingFaceBgeEmbeddings()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
VECTORSTORE_DIR_PATH = BASE_DIR / "data" / "vectorstore"


def get_vectorstore_path(user_id: str,thread_id:str,):
    return VECTORSTORE_DIR_PATH / user_id / thread_id


def create_vectorstore(user_id: str,thread_id:str, docs):
    path = get_vectorstore_path(user_id,thread_id)
    path.mkdir(parents=True, exist_ok=True)

    vectorstore = FAISS.from_documents(docs, embedding=embedding)
    vectorstore.save_local(str(path))
    return vectorstore


def load_vectorstore(user_id: str,thread_id:str):
    path = get_vectorstore_path(user_id,thread_id)

    if not path.exists():
        return None
    try:
        return FAISS.load_local(
            str(path),
            embedding,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"load error: {e}")
        return None

def update_vectorstore(docs, user_id: str,thread_id:str,file_hash:str):
    try:
        path = get_vectorstore_path(user_id,thread_id)
        vectorstore = load_vectorstore(user_id=user_id,thread_id=thread_id)

        if not docs:
            return False

        if vectorstore is None:
            create_vectorstore(user_id=user_id,thread_id=thread_id, docs=docs)
        else:
            vectorstore.add_documents(docs)
            vectorstore.save_local(str(path))

        return True

    except Exception as e:
        print(f"Vectorstore update error: {e}")
        return False



# def get_RetrievalQA(retriever):
#     return RetrievalQA.from_chain_type(llm=ChatGroq(model="llama-3.3-70b-versatile"),retriever=retriever)



# user langchain.chains.retrieval_qa.base import RetrievalQA
#RetrievalQA.from_chain_type(llm=llm,retriever=retriever)



# if __name__=="__main__":
#     print(str(VECTORSTORE_DIR_PATH))