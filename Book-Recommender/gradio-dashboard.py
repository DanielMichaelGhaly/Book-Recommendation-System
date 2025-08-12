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

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
)-> pd.DataFrame:
  recs = database_books.similarity_search(query, k = initial_top_k)
  books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
  books_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

  if category != "All":
   books_recs = books_recs[books_recs["simple_categories"] == category].head(final_top_k)
  else:
   books_recs = books_recs.head(final_top_k)

  if tone == "Happy":
   books_recs.sort_values(by = "joy", ascending=False, inplace = True)
  elif tone == "Surprising":
   books_recs.sort_values(by = "surprise", ascending = False, inplace = True)
  elif tone == "Angry":
   books_recs.sort_values(by = "anger", ascending = False, inplace = True)
  elif tone == "Suspense":
   books_recs.sort_values(by = "fear", ascending = False, inplace = True)
  elif tone == "Sad":
   books_recs.sort_values(by = "sadness", ascending = False, inplace = True)

  return books_recs

def recommend_books(
        query: str,
        category: str,
        tone: str
):

 recommendations = retrieve_semantic_recommendations(query, category, tone)
 results = []

 for _, row in recommendations.iterrows():
  description = row["description"]
  truncated_desc_split = description.split()
  truncated_description = " ".join(truncated_desc_split[:30]) + "..."

  authors_split = row["authors"].split(";")
  if len(authors_split) == 2:
   authors_str = f"{authors_split[0]} and {authors_split[1]}"
  elif len(authors_split)>2:
   authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
  else:
   authors_str = row["authors"]

  caption = f"{row['title']} by {authors_str}: {truncated_description}"
  results.append((row["large_thumbnail"], caption))
 return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspense", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
 # Title of dashboard
 gr.Markdown("Semantic book Recommender")
 with gr.Row():
  user_query = gr.Textbox(label = "Please enter a description of a book:",
                          placeholder = "e.g., A story about happiness")
  category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
  tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
  submit_button = gr.Button("Find recommendations")

 gr.Markdown("Recommendations")
 output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

 submit_button.click(fn = recommend_books,
                     inputs = [user_query, category_dropdown, tone_dropdown],
                     outputs = output)

if __name__ == "__main__":
 dashboard.launch()
