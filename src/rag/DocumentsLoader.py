from typing import Literal
from langchain_community.document_loaders import WebBaseLoader,PyMuPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
class DocLoader:
    def __init__(
            self,
            doctype:Literal["pdf","txt","url"],
            path,
            chunk_size:int =650,
            chunk_overlap:int=70):
        self.doctype = doctype
        self.path = path
        if not all([self.doctype,self.path]):
            raise ValueError("Please pass the required parameters [doctype,path]")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self):
        separators = ["\n\n", "\n", ".", " "]
        splitter =  RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=separators,
                is_separator_regex = True
            )
        try:
            if self.doctype in ["pdf","txt"]:
                if self.doctype =="pdf":
                    loader = PyMuPDFLoader(file_path=self.path)
                    doc = loader.load()
                elif self.doctype=="txt":
                    loader = TextLoader(file_path=self.path)
                    doc =  loader.load()
            elif self.doctype=="url":
                loader = WebBaseLoader(web_path=self.path)
                doc =   loader.load()

            chunked_docs = splitter.split_documents(doc)
            return chunked_docs

        except Exception as e:
            raise e