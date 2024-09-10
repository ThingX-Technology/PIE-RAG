import torch
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
import os
import chromadb

MODEL_NAME = 'zhangdah/phi-3-mini-LoRA'

directory_path = "memory storage"

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

Embedding = {}
Documents = {}
# Data Loading
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    category = filename.replace(".txt", "")
    with open(file_path, 'r') as file:
        text = file.read()
        Documents[category] = text
        Embedding[category] = embed_text(text)
    # category = filename.replace(".txt", "")
    # with open(file_path, 'r') as file:
    #     for line in file:
    #         input_ids, input_mask, segment_ids, labels = convert_line_to_inputs(line, category, label_map, tokenizer)
    #         processed_data['input_ids'].append(input_ids)
    #         processed_data['input_mask'].append(input_mask)
    #         processed_data['segment_ids'].append(segment_ids)
    #         processed_data['labels'].append(labels)


# embeddings = torch.stack([embedding_family, embedding_schedule, embedding_to_do_task])


# Initialize a Chroma client
client = chromadb.Client()

# Create a collection in the Chroma database
collection = client.create_collection(name="my_documents")

for topic in Embedding.keys():
    collection.add(
    embeddings=Embedding[topic].tolist(),  # Convert PyTorch tensor to numpy array
    documents=Documents[topic],
    ids = [topic]
)

# Store embeddings
# collection.add(
#     embeddings=embedding_family.tolist(),  # Convert PyTorch tensor to numpy array
#     documents=family_data,
#     ids = ['family']
# )

# collection.add(
#     embeddings=embedding_schedule.tolist(),  # Convert PyTorch tensor to numpy array
#     documents=schedule_data,
#     ids = ['schedule']
# )

# collection.add(
#     embeddings=embedding_to_do_task.tolist(),  # Convert PyTorch tensor to numpy array
#     documents=to_do_task_data,
#     ids = ['to_do_task']
# )

# Embed the query text
query_text = "Who is my mom?"
query_embedding = embed_text(query_text)

# Query the Chroma collection
results = collection.query(
    query_embeddings=query_embedding.numpy(),
    n_results=1  # Number of results to retrieve
)

# print(results)

# Initialize a text generation pipeline
generator = pipeline('text-generation', model='zhangdah/phi-3-mini-LoRA')

# Use the retrieved document to generate a response
context = results['documents'][0]  # Taking the top result for simplicity
generated_response = generator(f'''
                               Context: {context} 
                               Question: {query_text}
                               Answer:
                               ''', max_length = 1024)

print(generated_response[0]['generated_text'])