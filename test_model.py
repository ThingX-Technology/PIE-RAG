from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.bert.tokenization_bert import BertTokenizer
import torch


BERT_MODEL = "bert-base-uncased"
window_size = 15
step_size = 1

label_map = {'family': 0, 'schedule': 1, 'finance': 2, 'healthcare': 3, 'other': 4}
idx2label = {0: 'family', 1: 'schedule', 2: 'finance', 3: 'healthcare', 4: 'other'}
# Testing
model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = len(label_map))
model.load_state_dict(torch.load('tmp/auto_classifier.bin', map_location=torch.device('cpu')))
model.eval()

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

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

test_data = {'input_ids': [], 'input_mask': [], 'segment_ids': []}
# file_path = os.path.join(directory_path, 'test.txt')
with open('raw_text.txt', 'r') as file:
    for line in file:
        input_ids = tokenizer.encode(line, add_special_tokens = False)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        test_data ['input_ids'].append(input_ids)
        test_data ['input_mask'].append(input_mask)
        test_data ['segment_ids'].append(segment_ids)

print(test_data['input_ids'][0])
print(tokenizer.decode(test_data['input_ids'][0]))
input_ids, input_mask, segment_ids = slice_sequence(test_data['input_ids'], test_data['input_mask'], test_data['segment_ids'], None, window_size, step_size)

# test_input_ids, test_input_mask, test_segment_ids = get_data_loader(input_ids, input_mask, segment_ids, None, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=False)
test_input_ids = torch.tensor(input_ids, dtype=torch.long)
test_input_mask = torch.tensor(input_mask, dtype=torch.long)
test_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
outputs = model(test_input_ids, attention_mask=test_input_mask, token_type_ids=test_segment_ids)

logits = outputs.logits
predictions = torch.argmax(logits, dim=1)
print(predictions)

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

import os
for token_id, label in zip(input_ids, predictions):
    file_path = os.path.join('memory storage', idx2label[int(label)] + '.txt')
    token = tokenizer.decode(token_id)
    with open(file_path, 'w') as file:
        file.write(token)
