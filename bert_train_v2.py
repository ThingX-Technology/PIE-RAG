import torch
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
import os
from sklearn.model_selection import train_test_split
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.bert.tokenization_bert import BertTokenizer
import numpy as np
import tqdm

directory_path = 'dataset'

# label_map = {'B-family': 0, 'I-family': 1, 'B-schedule': 2, 'I-schedule': 3, 'B-finance': 4, 'I-finance': 5, 'B-healthcare': 6, 'I-healthcare': 7}
label_map = {'family': 0, 'schedule': 1, 'finance': 2, 'healthcare': 3, 'other': 4}

window_size = 20
step_size = 1
BERT_MODEL = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = len(label_map))
device = 'cpu'

MAX_SEQ_LENGTH=100


class BertInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, segment_ids, label_id):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        

def convert_line_to_inputs(line, label_type, label2idx, tokenizer, max_seq_length = None):
    """Loads a data file into a list of `InputBatch`s."""
    
    labels = []
    input_ids = tokenizer.encode(line, add_special_tokens = False)

    # BI label 
    # labels.append(label2idx[f"B-{label_type}"])
    # labels += len(input_ids[1:]) * [label2idx[f"I-{label_type}"]]

    # normal label
    labels += len(input_ids) * [label2idx[label_type]]

    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)
    # padding = [0] * (max_seq_length - len(input_ids))
    # input_ids += padding
    # input_mask += padding
    # segment_ids += padding
    return input_ids, input_mask, segment_ids, labels

processed_data = {'input_ids': [], 'input_mask': [], 'segment_ids': [], 'labels': []}

for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    category = filename.replace(".txt", "")
    with open(file_path, 'r') as file:
        for line in file:
            input_ids, input_mask, segment_ids, labels = convert_line_to_inputs(line, category, label_map, tokenizer)
            processed_data['input_ids'].append(input_ids)
            processed_data['input_mask'].append(input_mask)
            processed_data['segment_ids'].append(segment_ids)
            processed_data['labels'].append(labels)


# file_path = os.path.join(directory_path, 'family.txt')
# with open(file_path, 'r') as file:
#     for line in file:
#         input_ids, input_mask, segment_ids, labels = convert_line_to_inputs(line, 'family', label_map, tokenizer)
#         processed_data['input_ids'].append(input_ids)
#         processed_data['input_mask'].append(input_mask)
#         processed_data['segment_ids'].append(segment_ids)
#         processed_data['labels'].append(labels)

# file_path = os.path.join(directory_path, 'schedule.txt')
# with open(file_path, 'r') as file:
#     for line in file:
#         input_ids, input_mask, segment_ids, labels = convert_line_to_inputs(line, 'schedule', label_map, tokenizer)
#         processed_data['input_ids'].append(input_ids)
#         processed_data['input_mask'].append(input_mask)
#         processed_data['segment_ids'].append(segment_ids)
#         processed_data['labels'].append(labels)

# file_path = os.path.join(directory_path, 'finance.txt')
# with open(file_path, 'r') as file:
#     for line in file:
#         input_ids, input_mask, segment_ids, labels = convert_line_to_inputs(line, 'finance', label_map, tokenizer)
#         processed_data['input_ids'].append(input_ids)
#         processed_data['input_mask'].append(input_mask)
#         processed_data['segment_ids'].append(segment_ids)
#         processed_data['labels'].append(labels)

# file_path = os.path.join(directory_path, 'healthcare.txt')
# with open(file_path, 'r') as file:
#     for line in file:
#         input_ids, input_mask, segment_ids, labels = convert_line_to_inputs(line, 'healthcare', label_map, tokenizer)
#         processed_data['input_ids'].append(input_ids)
#         processed_data['input_mask'].append(input_mask)
#         processed_data['segment_ids'].append(segment_ids)
#         processed_data['labels'].append(labels)

# print(processed_data['input_ids'])
# print(processed_data['labels'])

train_input_ids, dev_input_ids, train_input_mask, dev_input_mask, train_segment_ids, dev_segment_ids, train_labels, dev_labels = train_test_split(processed_data['input_ids'], 
                                                                                                                        processed_data['input_mask'],
                                                                                                                        processed_data['segment_ids'],  
                                                                                                                        processed_data['labels'], test_size=0.2, random_state=1)

# print(train_input_ids)
# print(train_input_mask)
# print(train_segment_ids)
# print(train_labels)
# print(dev_input_ids)
# print(dev_input_mask)
# print(dev_segment_ids)
# print(dev_labels)

print("Train size:", len(train_input_ids))
print("Dev size:", len(dev_input_ids))
# print("Test size:", len(test_texts))

def slice_sequence(input_ids, input_mask, segment_ids, labels, window_size, step_size):

    proccessed_input_ids, proccessed_input_mask, proccessed_segment_ids, proccessed_train_labels = [], [], [], []

    if not labels:
        for i in range(len(input_ids)):
            input_ids[i] = [0] * (window_size//2) + input_ids[i] + [0] * (window_size//2)
            input_mask[i] = [1] * (window_size//2) + input_mask[i] + [1] * (window_size//2)
            segment_ids[i] = [0] * (window_size//2) + segment_ids[i] + [0] * (window_size//2)

            for j in range(0, len(input_ids[i]) - 2 * (window_size//2), step_size):
                proccessed_input_ids.append(input_ids[i][j: j + window_size])
                proccessed_input_mask.append(input_mask[i][j: j + window_size])
                proccessed_segment_ids.append(segment_ids[i][j: j + window_size])
        return proccessed_input_ids, proccessed_input_mask, proccessed_segment_ids

    else:
        for i in range(len(input_ids)):
            input_ids[i] = [0] * (window_size//2) + input_ids[i] + [0] * (window_size//2)
            input_mask[i] = [1] * (window_size//2) + input_mask[i] + [1] * (window_size//2)
            segment_ids[i] = [0] * (window_size//2) + segment_ids[i] + [0] * (window_size//2)

            for j in range(0, len(input_ids[i]) - 2 * (window_size//2), step_size):
                proccessed_input_ids.append(input_ids[i][j: j + window_size])
                proccessed_input_mask.append(input_mask[i][j: j + window_size])
                proccessed_segment_ids.append(segment_ids[i][j: j + window_size])
                proccessed_train_labels.append(labels[i][j])
        return proccessed_input_ids, proccessed_input_mask, proccessed_segment_ids, proccessed_train_labels

train_input_ids, train_input_mask, train_segment_ids, train_labels = slice_sequence(train_input_ids, train_input_mask, train_segment_ids, train_labels, window_size, step_size)
dev_input_ids, dev_input_mask, dev_segment_ids, dev_labels = slice_sequence(dev_input_ids, dev_input_mask, dev_segment_ids, dev_labels, window_size, step_size)

# print(dev_input_ids)
# print(dev_input_mask)
# print(dev_segment_ids)
# print(dev_labels)

print('Train Input ids size:', len(train_input_ids))
print('Train Input mask size:', len(train_input_mask))
print('Train segment ids size:', len(train_segment_ids))
print('Train label size:', len(train_labels))

print('Dev Input ids size:', len(dev_input_ids))
print('Dev Input mask size:', len(dev_input_mask))
print('Dev segment ids size:', len(dev_segment_ids))
print('Dev label size:', len(dev_labels))

# print(train_features)
# print(dev_features)

# Load data

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
def get_data_loader(input_ids, input_mask, segment_ids, labels, max_seq_length, batch_size, shuffle=True): 
    if not labels:
        all_input_ids = torch.tensor(input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
        return dataloader
    # print(len(input_ids[0]))
    else:
        all_input_ids = torch.tensor(input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        all_label_ids = torch.tensor(labels, dtype=torch.long)
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
        return dataloader
    

BATCH_SIZE = 32
# train_input_ids, train_input_mask, train_segment_ids, train_labels
# dev_input_ids, dev_input_mask, dev_segment_ids, dev_labels

train_dataloader = get_data_loader(train_input_ids, train_input_mask, train_segment_ids, train_labels, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=True)
dev_dataloader = get_data_loader(dev_input_ids, dev_input_mask, dev_segment_ids, dev_labels, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=False)

# # Evaluate
def evaluate(model, dataloader):
    model.eval()
    
    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

    for step, batch in enumerate(tqdm(dataloader, desc="Evaluation iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            # tmp_eval_loss, logits
            output= model(input_ids, attention_mask=input_mask,
                                          token_type_ids=segment_ids, labels=label_ids)

        tmp_eval_loss = output[0]
        logits = output[1]

        outputs = np.argmax(logits.to('cpu'), axis=1)
        label_ids = label_ids.to('cpu').numpy()
        
        predicted_labels += list(outputs)
        correct_labels += list(label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    
    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)
        
    return eval_loss, correct_labels, predicted_labels

# training
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
GRADIENT_ACCUMULATION_STEPS = 1
NUM_TRAIN_EPOCHS = 10
LEARNING_RATE = 1e-4
WARMUP_PROPORTION = 0.1
MAX_GRAD_NORM = 5

num_train_steps = int(len(train_dataloader.dataset) / BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(WARMUP_PROPORTION * num_train_steps)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, correct_bias=False)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

import torch
import os
from tqdm import trange
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support

OUTPUT_DIR = "tmp/"
MODEL_FILE_NAME = "auto_classifier.bin"
PATIENCE = 5

loss_history = []
no_improvement = 0
for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
    model.train()
    tr_loss = 0
    accuracy = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(tqdm(train_dataloader, desc="Training iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
        loss = outputs[0]

        if GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        loss.backward()
        tr_loss += loss.item()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)  
            
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
    dev_loss, correct_labels, predicted_labels = evaluate(model, dev_dataloader)

    accuracy = (predicted_labels == correct_labels).sum() / len(correct_labels)
    
    print("Loss history:", loss_history)
    print("Dev loss:", dev_loss)
    print("Dev Accuracy:", accuracy)
    
    
    if len(loss_history) == 0 or dev_loss < min(loss_history):
        no_improvement = 0
        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
    else:
        no_improvement += 1
    
    if no_improvement >= PATIENCE: 
        print("No improvement on development set. Finish training.")
        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        break        
    
    loss_history.append(dev_loss)


