from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world!", return_tensors="pt")
print(inputs)
outputs = model(**inputs)


from transformers import AutoTokenizer, AutoModel
import torch
device = 'cpu'
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

words = ["You are doshit!", "Iam typingAMESSisiasjdoasjdoasjd ", "Poliet man"]
sent_id = tokenizer.batch_encode_plus(words, padding=True, return_tensors="pt")
print(len(sent_id))
#input_ids = torch.Tensor(sent_id['input_ids'])
#attention_mask = torch.Tensor(sent_id['attention_mask'])
#print(input_ids, attention_mask)
outputs = model(**sent_id)
#print(outputs)
last_hidden_state = outputs['last_hidden_state']
pooler_output = outputs['pooler_output']
print(last_hidden_state.shape, pooler_output.shape)
#inputs = tokenizer(words, return_tensors="pt")
#outputs = model(input_ids, token_type_ids)
#print(outputs[1][0].shape)
