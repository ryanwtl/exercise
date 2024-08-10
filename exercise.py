import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import openai  # For generating embeddings (replace with your preferred model)

openai.api_key =  st.secrets["mykey"]

df = pd.read_csv("qa_dataset_with_embeddings.csv")

embeddings = np.array(df['Question_Embedding'].tolist())
def get_embedding(text):
  # Replace with your preferred embedding model
  response = openai.Embedding.create(
    input=[text],
    model="text-embedding-ada-002"
  )
  return response['data'][0]['embedding']
st.title("Smart Health FAQ")

user_question = st.text_input("Ask your health question:")
answer = st.empty()

if st.button("Submit"):
  if user_question:
    user_embedding = get_embedding(user_question)
    similarities = cosine_similarity(user_embedding.reshape(1, -1), embeddings).flatten()
    most_similar_index = similarities.argmax()
    max_similarity = similarities[most_similar_index]

    similarity_threshold = 0.7  # Adjust as needed

    if max_similarity >= similarity_threshold:
      answer.info(df.loc[most_similar_index, 'Answer'])
    else:
      answer.warning("I apologize, but I don't have information on that topic yet. Could you please ask other questions?")
  else:
    answer.warning("Please enter a question.")
st.button("Clear")
if st.button("Clear"):
  user_question = ""
  answer.empty()
answer.info(f"Similarity: {max_similarity:.2f}\n\n{df.loc[most_similar_index, 'Answer']}")
answer_rating = st.selectbox("How helpful was the answer?", ["Very Helpful", "Helpful", "Not Helpful"])
st.header("Common FAQs")
# Display a list of frequently asked questions
search_query = st.text_input("Search FAQs:")
# Filter the dataset based on the search query
