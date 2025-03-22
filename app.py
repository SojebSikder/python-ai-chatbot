import gradio as gr
import json
import faiss
from sentence_transformers import SentenceTransformer

# Load the sentence embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index and text data
index = faiss.read_index("medical_faiss_index.bin")

with open("medical_data.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

# Chatbot function


def chatbot(query):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=1)  # Get top-1 match

    if I[0][0] != -1:  # If a match is found
        return data[I[0][0]]["text"]
    else:
        return "I'm sorry, I don't have relevant information on that."


# print("Welcome to the Medical Chatbot! Type 'exit' to quit.")

# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["exit", "quit"]:
#         break
#     response = chatbot(user_input)
#     print("Bot:", response)

# Create a Gradio interface
iface = gr.Interface(fn=chatbot, inputs="text", outputs="text",
                     title="Medical Chatbot", description="Ask the chatbot any medical question!")

# Launch the interface
iface.launch()
