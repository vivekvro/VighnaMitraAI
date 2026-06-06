from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path
import asyncio
from dotenv import load_dotenv
load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
    )



async def postgres_embed(texts: list[str]) -> list[list[float]]:
    """
    True async wrapper around the blocking HuggingFaceEmbeddings.
    run_in_executor offloads the CPU work to a thread pool,
    keeping the asyncio event loop completely free.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,                        # uses default ThreadPoolExecutor
        embedding.embed_documents,   # the blocking sync method
        texts                        # argument to that method
    )

BASE_DIR = Path(__file__).resolve().parent.parent.parent
VECTORSTORE_DIR_PATH = BASE_DIR / "data" / "vectorstore"


def get_vectorstore_path(user_id: str,thread_id:str):
    return VECTORSTORE_DIR_PATH / user_id /thread_id


def create_vectorstore(user_id: str, docs,thread_id:str):
    path = get_vectorstore_path(user_id=user_id,thread_id=thread_id)

    vectorstore = FAISS.from_documents(docs, embedding=embedding)
    vectorstore.save_local(str(path))
    return vectorstore


def load_vectorstore(user_id: str,thread_id:str):
    path = get_vectorstore_path(user_id=user_id,thread_id=thread_id)

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

def update_vectorstore(docs, user_id: str,thread_id:str):
    if not docs:
            return False
    try:
        path = get_vectorstore_path(user_id=user_id,thread_id=thread_id)
        vectorstore = load_vectorstore(user_id=user_id,thread_id=thread_id)

        

        if vectorstore is None:
            create_vectorstore(user_id=user_id,thread_id=thread_id,docs=docs)
        else:
            vectorstore.add_documents(docs)
            vectorstore.save_local(str(path))

        return True

    except Exception as e:
        print(f"Vectorstore update error: {e}")
        return False
