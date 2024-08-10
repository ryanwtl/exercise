import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

openai.api_key =  st.secrets["mykey"]

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('/mnt/data/qa_dataset_with_embeddings.csv', on_bad_lines='skip')
    df['Question_Embedding'] = df['Question_Embedding'].apply(eval).apply(np.array)
    return df

# Select embedding model
@st.cache_resource
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

# Find the most relevant answer
def find_answer(user_question, df, model):
    user_embedding = model.encode([user_question])
    similarities = cosine_similarity(user_embedding, np.vstack(df['Question_Embedding'].values))
    max_similarity_idx = np.argmax(similarities)
    max_similarity_score = similarities[0, max_similarity_idx]
    
    threshold = 0.75  # Experiment with this threshold
    if max_similarity_score > threshold:
        return df.iloc[max_similarity_idx]['Answer'], max_similarity_score
    else:
        return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?", None

# Streamlit interface
st.title("Health Q&A Chatbot")
st.write("Ask questions about heart, lung, and blood-related health topics.")

df = load_data()
model = load_model()

user_question = st.text_input("Enter your question here:")
if st.button("Get Answer"):
    if user_question:
        answer, score = find_answer(user_question, df, model)
        st.write("**Answer:**", answer)
        if score:
            st.write("**Similarity Score:**", score)
    else:
        st.write("Please enter a question.")

if st.button("Clear"):
    st.experimental_rerun()
