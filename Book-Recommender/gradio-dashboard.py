import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(books["large_thumbnail"].isna(), "cover-not-found.jpg", books["large_thumbnail"])

raw_documents = TextLoader("tagged_description.txt", encoding = "utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

emb = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2",
 encode_kwargs={"normalize_embeddings": True}
)

database_books= Chroma.from_documents(documents, embedding=emb)