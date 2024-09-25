import torch

from executorch.exir import EdgeCompileConfig, to_edge
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch._export import capture_pre_autograd_graph
from torch.export import export

# from model import GPT
from transformers.models.bert.modeling_bert import BertForSequenceClassification

# Load the model.
# model = GPT.from_pretrained('gpt2')
# tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels = len(self.label_map))
model.load_state_dict(torch.load('tmp/auto_classifier.bin', map_location=torch.device('cpu')))

# Create example inputs. This is used in the export process to provide
# hints on the expected shape of the model input.
example_inputs = (torch.randint(0, 100, (1, model.config.block_size), dtype=torch.long), )

# Set up dynamic shape configuration. This allows the sizes of the input tensors
# to differ from the sizes of the tensors in `example_inputs` during runtime, as
# long as they adhere to the rules specified in the dynamic shape configuration.
# Here we set the range of 0th model input's 1st dimension as
# [0, model.config.block_size].
# See https://pytorch.org/executorch/main/concepts.html#dynamic-shapes
# for details about creating dynamic shapes.
dynamic_shape = (
    {1: torch.export.Dim("token_dim", max=model.config.block_size)},
)

# Trace the model, converting it to a portable intermediate representation.
# The torch.no_grad() call tells PyTorch to exclude training-specific logic.
with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
    m = capture_pre_autograd_graph(model, example_inputs, dynamic_shapes=dynamic_shape)
    traced_model = export(m, example_inputs, dynamic_shapes=dynamic_shape)

# Convert the model into a runnable ExecuTorch program.
edge_config = EdgeCompileConfig(_check_ir_validity=False)
edge_manager = to_edge(traced_model,  compile_config=edge_config)
et_program = edge_manager.to_executorch()

# Save the ExecuTorch program to a file.
with open("bert_classify.pte", "wb") as file:
    file.write(et_program.buffer)