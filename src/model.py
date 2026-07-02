import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForSeq2SeqLM

class FallacyDetectionModel(nn.Module):
    """
    DeBERTa-based model for classifying logical fallacies.
    """
    def __init__(self, model_name="microsoft/deberta-v3-base", num_labels=9):
        super(FallacyDetectionModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

class FallacyGenerationModel(nn.Module):
    """
    GPT-2 or T5 based model for generating logically consistent or fallacious arguments.
    """
    def __init__(self, model_name="gpt2", model_type="causal"):
        super(FallacyGenerationModel, self).__init__()
        self.model_type = model_type
        if model_type == "causal":
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        elif model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            raise ValueError("model_type must be 'causal' or 'seq2seq'")
            
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def generate(self, input_ids, attention_mask, **kwargs):
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
