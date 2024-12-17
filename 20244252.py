import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
from datasets import load_dataset
import copy
import os
import torch.nn.functional as F
import argparse
import random
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW
from pathlib import Path
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


# Constants
CIFAR_BATCH_SIZE = 128
LM_BATCH_SIZE = 32
VL_BATCH_SIZE = 16
MAX_LENGTH = 128
HIDDEN_SIZE = 768 #
NUM_EPOCHS = 1 #
IMG_PATCH = '<img>' #
NUM_IMG_TOKEN = 32 # 
VLM_MAX_LENGTH = 32 #



def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# CIFAR-10 Dataset and DataLoader
def get_cifar10_loaders():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=CIFAR_BATCH_SIZE,
                           shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=CIFAR_BATCH_SIZE,
                          shuffle=False, num_workers=2)
    
    return trainloader, testloader

# ELI5 Dataset
class ELI5Dataset(Dataset):
    def __init__(self,tokenizer, MAX_POSITION_EMBEDDINGS, data_type):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.block_size = MAX_POSITION_EMBEDDINGS
        
        if data_type == "train":
            data = load_dataset("eli5_category", split="train[:3000]", trust_remote_code=True)
            data = data.select(range(1000))
        elif data_type == "valid":
            data = load_dataset("eli5_category", split="validation1[:2000]", trust_remote_code=True)
        elif data_type == "test":
            data = load_dataset("eli5_category", split="test[:20]", trust_remote_code=True)

        data = data.flatten() 
        data = data.map(self.preprocess_function, batched=True,num_proc=8,remove_columns=data.column_names)
        data = data.map(self.group_texts, batched=True, num_proc=8)
        result =[]
        for i in data:
            result.append(i['input_ids'])
        self.final_data = torch.tensor(result).to(torch.int64)
        
    def preprocess_function(self, examples):
        return self.tokenizer([" ".join(x) for x in examples["answers.text"]])
    
    def group_texts(self, examples):

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]]) 
        if total_length >= (self.block_size-2):
            total_length = (total_length // (self.block_size-2)) * (self.block_size-2)
        result = {
            k: [[self.tokenizer.bos_token_id]+t[i : i + self.block_size-2]+[self.tokenizer.eos_token_id] for i in range(0, total_length, self.block_size-2)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
        
    def __len__(self):
        return len(self.final_data)
    
    def __getitem__(self, idx):
        return self.final_data[idx]

# LLaVA Dataset
def transform_fn(is_train):
    if is_train:
        return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    else:
        return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Do not change
class LLaVADataset(Dataset):
    def __init__(self, json_file, img_path, tokenizer, is_train):
        super().__init__()

        self.transform = transform_fn(is_train)

        self.json_file = json_file

        self.tokenizer = tokenizer
        self.img_path = img_path

        self.ignore_idx = -100
        self.begin_signal = tokenizer.bos_token
        self.end_signal = tokenizer.eos_token

        with open(self.json_file) as json_file:
            data = json.load(json_file)

        if is_train:
            data = data[:1000]
        else:
            data = data[1000:]

        self.data = data

    def preprocess(self, conversation):
        question = self.begin_signal + "human: " + conversation[0]['value'] + self.end_signal
        answer = self.begin_signal + "assistant: " + conversation[1]['value'] + self.end_signal

        tokenized_q = self.tokenizer(question, return_tensors="pt")

        combined_qa = question + answer
        tokenized_qa = self.tokenizer(combined_qa, padding="max_length", truncation=True,
                                      max_length=VLM_MAX_LENGTH, return_tensors="pt")

        input_ids = tokenized_qa.input_ids[0]
        label = copy.deepcopy(input_ids)
        len_of_q = len(tokenized_q.input_ids[0])
        label[:len_of_q] = self.ignore_idx

        len_of_pad = tokenized_qa.input_ids.eq(self.tokenizer.pad_token_id).sum().item()
        label[-len_of_pad:] = self.ignore_idx

        return input_ids, label
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        meta = self.data[idx]

        image_id = meta['image']
        image = Image.open(os.path.join(self.img_path, image_id)).convert('RGB')
        image = self.transform(image)

        conversation = meta['conversation']
        input_id, label = self.preprocess(conversation)

        return dict(image=image, input_ids=input_id, label=label)







"""
MODELS
"""
class ResNet18_CIFAR10(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = torchvision.models.resnet18(num_classes=10) # resnet18

    def forward(self, x):
        x = self.model(x)
        return x







class CustumGPT(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        config = GPT2Config(
            n_positions=MAX_LENGTH,
            n_ctx=MAX_LENGTH,
            n_embd=HIDDEN_SIZE,
        )
        self.model = GPT2LMHeadModel(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)





class VisionLanguageModel(torch.nn.Module):
    def __init__(self, vision_encoder, language_decoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_decoder = language_decoder
        self.image_to_text = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, images, input_ids, attention_mask=None, labels=None):
        vision_features = self.vision_encoder(images)
        vision_features = self.image_to_text(vision_features)
        input_embeddings = self.language_decoder.model.transformer.wte(input_ids)
        input_embeddings[:, :NUM_IMG_TOKEN, :] = vision_features.unsqueeze(1).expand(-1, NUM_IMG_TOKEN, -1)
        return self.language_decoder(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

"""
End of MODELS
"""







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='./instruct_tuning/instruct.json')
    parser.add_argument('--image_folder_path', type=str, default='./instruct_tuning/images/')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(IMG_PATCH, special_tokens=True)
    
    # cifar10
    cifar_trainloader, cifar_testloader = get_cifar10_loaders()
    
    # eli5
    eli5_dataset = ELI5Dataset(tokenizer, MAX_LENGTH, 'train')
    eli5_loader = DataLoader(eli5_dataset, batch_size=LM_BATCH_SIZE, shuffle=True)
    
    # llava train
    llava_dataset = LLaVADataset(args.json_path, args.image_folder_path, tokenizer, is_train=True)
    llava_loader = DataLoader(llava_dataset, batch_size=VL_BATCH_SIZE, shuffle=True)
    
    # llava test
    test_llava_dataset = LLaVADataset(args.json_path, args.image_folder_path, tokenizer, is_train=False)
    test_llava_loader = DataLoader(test_llava_dataset, batch_size=VL_BATCH_SIZE, shuffle=False)
    




    
    ########################
    ##### Vision Encoder ###
    ########################
    print("Start training Vision Encoder....")
    
    resnet_model = ResNet18_CIFAR10().to(device)
    optimizer = torch.optim.AdamW(resnet_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 2
    
    resnet_model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(cifar_trainloader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = resnet_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(cifar_trainloader):.4f}")
    
    
    
    
    
    
    
    
    #########################
    #### Text Decoder #######
    #########################
    print("Start training Text Decoder....")

    text_decoder = CustumGPT(tokenizer).to(device)
    lm_optimizer = torch.optim.AdamW(text_decoder.parameters(), lr=1e-3)   
    epochs = 1
    
    text_decoder.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in tqdm(eli5_loader):
            input_ids = batch.to(device)
            labels = input_ids.clone()
            
            lm_optimizer.zero_grad()
            outputs = text_decoder(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            lm_optimizer.step()
            running_loss += loss.item()
    
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(eli5_loader):.4f}")
 
    
    
    
    
    
    
    
    #####################
    ####### VLM #########
    #####################
    print("Start training VLM....")
    
    # Change fc layer of ResNet
    resnet_model.model.fc = torch.nn.Linear(resnet_model.model.fc.in_features, HIDDEN_SIZE)
    vision_encoder = resnet_model
    
    
    vl_model = VisionLanguageModel(vision_encoder, text_decoder).to(device)
    vl_optimizer = torch.optim.AdamW(vl_model.parameters(), lr=1e-4)
    epochs = 5
    vl_model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in tqdm(llava_loader):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            vl_optimizer.zero_grad()
            outputs = vl_model(images, input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            vl_optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(llava_loader):.4f}")







    #####################
    ####### VLM #########
    #####################
    print("Evaluating VLM....")
    
    vl_model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in tqdm(test_llava_loader):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)

            outputs = vl_model(images, input_ids)
            logits = outputs.logits.cpu().numpy()
            all_logits.append(logits)
            
    all_logits = np.concatenate(all_logits, axis=0)
    
    save_path = '20244252.npy'
    np.save(save_path, all_logits)
    print(f"Logits saved to {save_path}.")
    

if __name__ == "__main__":
    main()