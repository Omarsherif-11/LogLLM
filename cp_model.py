import os.path
import peft
import torch
from transformers import BertTokenizerFast, BertModel, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, DynamicCache
import numpy as np
from torch import nn
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType, set_peft_model_state_dict, get_peft_model_state_dict

# ... [Previous Imports and Helper Functions merge_data, stack_and_pad_right, stack_and_pad_left remain unchanged] ...
def merge_data(data):
    merged_data = []
    start_positions = []
    current_position = 0
    for sublist in data:
        start_positions.append(current_position)
        merged_data.extend(sublist)
        current_position += len(sublist)
    return merged_data, start_positions

def stack_and_pad_right(tensors):
    max_len = max(tensor.shape[0] for tensor in tensors)
    padded_tensors = []
    padding_masks = []
    for tensor in tensors:
        pad_len = max_len - tensor.shape[0]
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_len))
        padded_tensors.append(padded_tensor)
        padding_mask = torch.cat([torch.ones(tensor.shape[0], dtype=torch.long),
                                  torch.zeros(pad_len, dtype=torch.long)])
        padding_masks.append(padding_mask)
    stacked_tensor = torch.stack(padded_tensors)
    padding_masks = torch.stack(padding_masks)
    return stacked_tensor, padding_masks

def stack_and_pad_left(tensors):
    max_len = max(tensor.shape[0] for tensor in tensors)
    padded_tensors = []
    padding_masks = []
    for tensor in tensors:
        pad_len = max_len - tensor.shape[0]
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, pad_len, 0))
        padded_tensors.append(padded_tensor)
        padding_mask = torch.cat([torch.zeros(pad_len, dtype=torch.long),
                                 torch.ones(tensor.shape[0], dtype=torch.long)])
        padding_masks.append(padding_mask)
    stacked_tensor = torch.stack(padded_tensors)
    padding_masks = torch.stack(padding_masks)
    return stacked_tensor, padding_masks

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

class LogLLM(nn.Module):
    def __init__(self, Bert_path, Llama_path, ft_path=None, is_train_mode=True, device = torch.device("cuda:0"), max_content_len = 128, max_seq_len = 128):
        super().__init__()
        self.max_content_len = max_content_len
        self.max_seq_len = max_seq_len
        self.device = device
        self.Llama_tokenizer = AutoTokenizer.from_pretrained(Llama_path, padding_side="right")
        self.Llama_tokenizer.pad_token = self.Llama_tokenizer.eos_token
        self.Llama_model = AutoModelForCausalLM.from_pretrained(Llama_path, quantization_config=bnb_config,
                                                           low_cpu_mem_usage=True,
                                                           device_map=device)

        self.Bert_tokenizer = BertTokenizerFast.from_pretrained(Bert_path, do_lower_case=True)
        self.Bert_model = BertModel.from_pretrained(Bert_path, quantization_config=bnb_config, low_cpu_mem_usage=True,
                                               device_map=device)

        self.projector = nn.Linear(self.Bert_model.config.hidden_size, self.Llama_model.config.hidden_size, device=device)

        self.instruc_tokens = self.Llama_tokenizer(
            ['Below is a sequence of system log messages:', '. Is this sequence normal or anomalous? \\n'],
            return_tensors="pt", padding=True).to(self.device)

        # Logic for initialization (Fresh or Pre-trained adapters)
        if ft_path is not None and os.path.exists(os.path.join(ft_path, 'Llama_ft')):
            print(f'Loading peft model from {ft_path}.')
            Llama_ft_path = os.path.join(ft_path, 'Llama_ft')
            Bert_ft_path = os.path.join(ft_path, 'Bert_ft')
            projector_path = os.path.join(ft_path, 'projector.pt')
            
            self.Llama_model = PeftModel.from_pretrained(
                self.Llama_model,
                Llama_ft_path,
                is_trainable=is_train_mode,
                torch_dtype=torch.float16,
            )
            self.Bert_model = PeftModel.from_pretrained(
                self.Bert_model,
                Bert_ft_path,
                is_trainable=is_train_mode,
                torch_dtype=torch.float16,
            )
            # Load projector state
            if os.path.exists(projector_path):
                self.projector.load_state_dict(torch.load(projector_path, map_location=device, weights_only=True))
        else:
            print(f'Creating new peft adapters.')
            Bert_peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                                          r=4,
                                          lora_alpha=32,
                                          lora_dropout=0.01)
            self.Bert_model = get_peft_model(self.Bert_model, Bert_peft_config)

            Llama_peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.Llama_model = get_peft_model(self.Llama_model, Llama_peft_config)

    def save_ft_model(self, path):
        """Final export save"""
        if not os.path.exists(path):
            os.makedirs(path)
        Llama_ft_path = os.path.join(path,'Llama_ft')
        Bert_ft_path = os.path.join(path,'Bert_ft')
        projector_path = os.path.join(path,'projector.pt')
        self.Llama_model.save_pretrained(Llama_ft_path, safe_serialization = True)
        self.Bert_model.save_pretrained(Bert_ft_path, safe_serialization =True)
        torch.save(self.projector.state_dict(), projector_path)

    def save_checkpoint(self, path):
        """Intermediate checkpoint save of trainable parameters only"""
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save Adapters using PEFT's native state dict retrieval
        llama_state = get_peft_model_state_dict(self.Llama_model)
        bert_state = get_peft_model_state_dict(self.Bert_model)
        projector_state = self.projector.state_dict()
        
        torch.save({
            'llama_adapter': llama_state,
            'bert_adapter': bert_state,
            'projector': projector_state
        }, os.path.join(path, 'model_weights.pt'))

    def load_checkpoint(self, path):
        """Load intermediate checkpoint"""
        weights_path = os.path.join(path, 'model_weights.pt')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Checkpoint weights not found at {weights_path}")
            
        print(f"Loading model weights from {weights_path}...")
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # Load Adapters
        set_peft_model_state_dict(self.Llama_model, checkpoint['llama_adapter'])
        set_peft_model_state_dict(self.Bert_model, checkpoint['bert_adapter'])
        
        # Load Projector
        self.projector.load_state_dict(checkpoint['projector'])

    def set_train_only_projector(self):
        for name, param in self.projector.named_parameters():
            param.requires_grad = True
        for name, param in self.Bert_model.named_parameters():
            param.requires_grad = False
        for name, param in self.Llama_model.named_parameters():
            param.requires_grad = False

    def set_train_only_Llama(self):
        for name, param in self.projector.named_parameters():
            param.requires_grad = False
        for name, param in self.Bert_model.named_parameters():
            param.requires_grad = False
        for name, param in self.Llama_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

    def set_train_projectorAndBert(self):
        for name, param in self.projector.named_parameters():
            param.requires_grad = True
        for name, param in self.Bert_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
        for name, param in self.Llama_model.named_parameters():
            param.requires_grad = False

    def set_finetuning_all(self):
        for name, param in self.projector.named_parameters():
            param.requires_grad = True
        for name, param in self.Bert_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
        for name, param in self.Llama_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

    def train_helper(self, inputs, seq_positions, labels):
        # [Content remains exactly as in your provided file]
        # I am omitting the body here for brevity as it was not requested to be changed, 
        # but in your actual file, keep the exact logic you provided for train_helper.
        batch_size = len(labels)
        outputs = self.Bert_model(**inputs).pooler_output
        outputs = outputs.float()
        outputs = self.projector(outputs)
        outputs = outputs.half()
        seq_embeddings = torch.tensor_split(outputs, seq_positions)
        prefix = "The sequence is "
        max_len = max(len(s) for s in labels) + len(prefix)
        labels = np.char.add(np.char.add(prefix, labels.astype(f'U{max_len}')), ".")
        answer_tokens = self.Llama_tokenizer(list(labels), padding=True, return_tensors="pt").to(self.device)
        target_tokens_ids = torch.cat([answer_tokens['input_ids'][:, 1:],
                                       torch.full((batch_size, 1), self.Llama_tokenizer.eos_token_id, device=self.device)],
                                      dim=-1)
        target_tokens_atts = answer_tokens['attention_mask'].bool()
        answer_tokens_ids = answer_tokens['input_ids'][:, 1:]
        answer_tokens_atts = answer_tokens['attention_mask'].bool()[:, 1:]
        
        # Handle Peft vs Standard nesting
        if isinstance(self.Llama_model, peft.peft_model.PeftModelForCausalLM):
            # Access internal model for embeddings
            base_model = self.Llama_model.model if hasattr(self.Llama_model, 'model') else self.Llama_model
            # Depending on peft version, might be model.model.embed_tokens or just model.embed_tokens
            # Your code used self.Llama_model.model.model.embed_tokens
            instruc_embeddings = base_model.model.embed_tokens(self.instruc_tokens['input_ids'])
            answer_embeddings = base_model.model.embed_tokens(answer_tokens_ids)
        else:
            instruc_embeddings = self.Llama_model.model.embed_tokens(self.instruc_tokens['input_ids'])
            answer_embeddings = self.Llama_model.model.embed_tokens(answer_tokens_ids)

        ins1 = instruc_embeddings[0][self.instruc_tokens['attention_mask'][0].bool()]
        ins2 = instruc_embeddings[1][self.instruc_tokens['attention_mask'][1].bool()][1:]

        embeddings = []
        target_lens = []
        for seq_embedding, answer_embedding, answer_tokens_att in zip(seq_embeddings, answer_embeddings,
                                                                      answer_tokens_atts):
            full_prompt_embedding = torch.cat([ins1, seq_embedding, ins2, answer_embedding[answer_tokens_att]])
            target_lens.append(answer_tokens_att.sum())
            embeddings.append(full_prompt_embedding)

        inputs_embeds, attention_mask = stack_and_pad_left(embeddings)
        attention_mask = attention_mask.to(self.device)
        label_mask = attention_mask.clone()
        for i in range(label_mask.shape[0]):
            label_mask[i, :-target_lens[i]-1] = 0
        label_mask = label_mask.bool()

        Llama_output = self.Llama_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits

        return Llama_output[label_mask], target_tokens_ids[target_tokens_atts]

    # [Keep forward method exactly as provided in original]
    def forward(self, inputs, seq_positions):
         # ... (Use your original code here)
         pass