from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModel
import numpy as np
# import chromadb
# from classify_topic import TopicClassifier
import os
# import torch

# requirements for buildozer
# requirements = python3,kivy,libb2,openssl,transformers,chromadb,torch,huggingface_hub,tqdm,pyyaml,filelock,packaging,cython,cython3,numpy,certifi,requests

class MainScreen(GridLayout):

    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.cols = 2  # Two columns for label and input fields
        
        # chromadb Vector Database setup
        # self.embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        # self.embedding_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.dimension = 128  # Dimensions of sentence transformer embeddings
        self.embedding = {}
        self.documents = {}
        # self.chromadb_client = chromadb.Client()
        # self.chromadb_collection = self.chromadb_client.create_collection(name="my_documents")

        # Block 1: Text Input for importing data
        self.add_widget(Label(text='Import Text'))
        self.input_text = TextInput(multiline=True)
        self.add_widget(self.input_text)
        
        self.import_button = Button(text="Import Text")
        self.import_button.bind(on_press=self.import_text_to_db)
        self.add_widget(self.import_button)

        # Block 2: Text Input for query
        # self.add_widget(Label(text='Query Text'))
        # self.query_text = TextInput(multiline=True)
        # self.add_widget(self.query_text)
        #
        # self.run_button = Button(text="Run Query")
        # self.run_button.bind(on_press=self.run_query)
        # self.add_widget(self.run_button)
        #
        # # Block 3: To display results
        # self.result_label = Label(text="Result will be displayed here.")
        # self.add_widget(self.result_label)
    
    def import_text_to_db(self, instance):
        """
        Takes the user input text, encodes it using the sentence transformer,
        and stores it in the chromadb vector database.
        """

        memory_dir = "memory_storage"
        source_file = "test.txt" # user input will be stored here

        input_text = self.input_text.text.strip()
        with open(source_file, 'w') as file:
            file.write(input_text)
        
        # model = TopicClassifier(source_file, model_name = "bert-base-uncased", tokenizer = "bert-base-uncased", window_size = 15, step_size = 1)
        # topic2text_dict = model.classify()
        # model.save_to_memory(topic2text_dict, memory_dir)

        for filename in os.listdir(memory_dir):
            file_path = os.path.join(memory_dir, filename)
            category = filename.replace(".txt", "")
            with open(file_path, 'r') as file:
                text = file.read()
                self.documents[category] = text
                inputs = self.embedding_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
                # with torch.no_grad():
                #     self.embedding[category] = self.embedding_model(**inputs).last_hidden_state.mean(dim=1)

        # for topic in self.embedding.keys():
        #     self.chromadb_collection.add(
        #     embeddings=self.embedding[topic].tolist(),  # Convert PyTorch tensor to numpy array
        #     documents=self.documents[topic],
        #     ids = [topic]
        # )


    def run_query(self, instance):
        """
        Takes the user's query, retrieves the closest match from the FAISS vector store,
        and combines it with the query.
        """
        
        # Embed the query text
        query = self.query_text.text.strip()
        inputs = self.embedding_tokenizer(query, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            query_embedding = self.embedding_model(**inputs).last_hidden_state.mean(dim=1)

        # Query the Chroma collection
        results = self.chromadb_collection.query(
            query_embeddings=query_embedding.numpy(),
            n_results=1  # Number of results to retrieve
        )
        context = results['documents'][0][0]

        prompt = f'''
                Context: {context} 
                Question: {query}
                Answer:
                '''
        
        self.result_label.text = prompt

class Test(GridLayout):

    def __init__(self, **kwargs):
        super(Test, self).__init__(**kwargs)
        self.cols = 2
        self.add_widget(Label(text='Import Text'))
        self.input_text = TextInput(multiline=True)
        self.add_widget(self.input_text)

class MyApp(App):
    # def build(self):
    #     return MainScreen()
    def build(self):
        return Test()

if __name__ == '__main__':
    MyApp().run()
