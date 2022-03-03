import torch

from transformers import BertTokenizer, BertModel


class ChineeseBert:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.model = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext").to(self.device)
        self.model.eval()

    def get_bert_embedding(self, lines):
        inputs = self.tokenizer(lines, return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            output = self.model(**inputs, output_hidden_states=True)
        last_embedding = output.hidden_states[-1].cpu().numpy()
        return last_embedding
