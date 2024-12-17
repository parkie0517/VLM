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
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from torch.optim.lr_scheduler import StepLR




# Constants
CIFAR_BATCH_SIZE = 128
LM_BATCH_SIZE = 32
VL_BATCH_SIZE = 16
MAX_LENGTH = 128
HIDDEN_SIZE = 768
NUM_EPOCHS = 1
IMG_PATCH = '<img>'
NUM_IMG_TOKEN = 32
VLM_MAX_LENGTH = 32

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
MODEL
"""

# 2. Define the ResNet-18 Model
class ResNet18_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_CIFAR10, self).__init__()
        self.model = models.resnet18(pretrained=False)
        
        # Modify the first convolution for 32x32 images
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove the pooling layer
        
        # Replace the final fully connected layer for CIFAR-10 classes
        self.model.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    
# Step 1: Vision Encoder (ResNet18 on CIFAR-10)
class VisionEncoder(nn.Module):
    def __init__(self, resnet18, hidden_size):
        super(VisionEncoder, self).__init__()
 

        # Remove classification head
        resnet18.fc = nn.Identity()
        self.model = resnet18
        
        self.fc = nn.Linear(512, hidden_size)  # Map to GPT-2 input size

    def forward(self, x):

        features = self.model(x)  # [batch, 512]
        return self.fc(features).unsqueeze(1)  # [batch, 1, HIDDEN_SIZE]


# Step 2: Text Decoder (GPT-2)
class VisionLanguageModel(nn.Module):
    def __init__(self, vision_encoder, text_decoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        

    def forward(self, images, input_ids, attention_mask):
        vision_tokens = self.vision_encoder(images)  # [batch, 1, HIDDEN_SIZE]

        # Append visual tokens at the start of the input sequence
        input_embeddings = self.text_decoder.transformer.wte(input_ids)
        embeddings = torch.cat([vision_tokens, input_embeddings], dim=1)

        outputs = self.text_decoder(inputs_embeds=embeddings, attention_mask=attention_mask)
        return outputs
    
"""
END of MODEL
"""



"""
TRAIN Vision Encoder
"""
def train_vision(epoch, model, trainloader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:  # Print loss every 100 batches
            print(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / 100:.3f}")
            running_loss = 0.0


# 5. Testing the Model
def test_vision(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")
"""
End of TRAIN Vision Encoder
"""







"""
TRAIN GPT Decoder
"""
# Define the training function
def train_gpt2(text_decoder, dataloader, optimizer, criterion, device, epochs=3):
    text_decoder.train()
    
    for epoch in range(epochs):
        total_loss = 0
        print(f"Epoch {epoch + 1}/{epochs}")
        
        for step, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch.to(device)  # Input IDs from ELI5 dataset
            labels = input_ids.clone()  # GPT-2 predicts next tokens
            
            # Zero out gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = text_decoder(input_ids=input_ids, labels=labels)
            loss = outputs.loss  # Loss is automatically computed for CLM
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Print loss every 10 steps
            if (step + 1) % 10 == 0:
                print(f"Step {step + 1}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

"""
End of TRAIN GPT Decoder
"""







"""
TRAIN VLM
"""
# Training function
def train_vlm(model, tokenizer, dataloader, optimizer, criterion, scheduler ,device, epoch):
    model.train()
    for images, input_ids, labels in dataloader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)
        
        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")


def test_vlm(model, tokenizer, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, input_ids, labels in dataloader:
            images, input_ids, labels = images.to(device), input_ids.to(device), labels.to(device)
            attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)
"""
End of TRAIN VLM
"""


def save_logits(model, tokenizer, data_loader, device, LOGITS_FILE):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)
            outputs = model(images, input_ids, attention_mask)
            all_logits.append(outputs.logits.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    np.save(LOGITS_FILE, all_logits)
    print(f"Logits saved to {LOGITS_FILE}")

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
    print("Training ResNet18 on CIFAR-10...")
    model_resnet18 = ResNet18_CIFAR10().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_resnet18.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    num_epochs = 20
    for epoch in range(num_epochs):
        train_vision(epoch, model_resnet18, cifar_trainloader, device, optimizer, criterion)
        test_vision(model_resnet18, cifar_testloader, device)
    
    # Define paths to save the model and optimizer
    MODEL_PATH = "resnet18_cifar10.pth"
    OPTIMIZER_PATH = "optimizer_resnet18.pth"

    # Save model state
    # just in case i scew up, im gonna save a checkpoint
    torch.save(model_resnet18.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Save optimizer state
    torch.save(optimizer.state_dict(), OPTIMIZER_PATH)
    print(f"Optimizer saved to {OPTIMIZER_PATH}")
    
    # Initialize the model
    
    model_resnet18_copy = ResNet18_CIFAR10().to(device)
    model_resnet18.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model_resnet18.eval()  # Set model to evaluation mode
    print(f"Model loaded from {MODEL_PATH}")
    
    print("Created VisionEncoder based on ResNet18...")
    vision_encoder = VisionEncoder(model_resnet18, HIDDEN_SIZE).to(device)
    
    
    
    
    
    
    
    
    
    #########################
    #### Text Decoder #######
    #########################
    print("Training Text Decoder on ELI5...")
    text_decoder = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    text_decoder.resize_token_embeddings(len(tokenizer))  # Adjust for any new tokens if added


    optimizer = AdamW(text_decoder.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Train the model
    train_gpt2(text_decoder, eli5_loader, optimizer, criterion, device)
    
    
    
    
    
    
    #####################
    ####### VLM #########
    #####################
    print("Fine-tuning Vision-Language Model...")

    model = VisionLanguageModel(vision_encoder, text_decoder).to(device)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.vision_encoder.fc.parameters():
        param.requires_grad = True
    optimizer_vlm = optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = StepLR(optimizer_vlm, step_size=1, gamma=0.95)
    criterion_vlm = nn.CrossEntropyLoss(ignore_index=-100)
    
    for epoch in range(NUM_EPOCHS):
        train_vlm(model, tokenizer, llava_loader, optimizer_vlm, criterion_vlm, scheduler, device, epoch)
        test_vlm(model, tokenizer, test_llava_loader, criterion, device)
    
    
    # Save logits
    LOGITS_FILE = "20244252.npy"
    save_logits(model, tokenizer, test_llava_loader, device, LOGITS_FILE)
    
    
if __name__ == "__main__":
    main()