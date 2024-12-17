import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import os
from studentID import get_cifar10_loaders, LLaVADataset, ELI5Dataset



# Constants
NUM_EPOCHS = 1
NUM_IMG_TOKEN = 32
HIDDEN_SIZE = 768
LOGITS_FILE = "20244252.npy"

# Step 1: Vision Encoder (ResNet18 on CIFAR-10)
class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet18 = models.resnet18(pretrained=False)
        resnet18.fc = nn.Identity()  # Remove classification head
        self.backbone = resnet18
        self.fc = nn.Linear(512, HIDDEN_SIZE)  # Map to GPT-2 input size

    def forward(self, x):
        features = self.backbone(x)  # [batch, 512]
        return self.fc(features).unsqueeze(1)  # [batch, 1, HIDDEN_SIZE]

# Step 2: Text Decoder (GPT-2)
class VisionLanguageModel(nn.Module):
    def __init__(self, vision_encoder, text_decoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.linear_layer = nn.Linear(HIDDEN_SIZE, text_decoder.config.n_embd)

    def forward(self, images, input_ids, attention_mask):
        vision_features = self.vision_encoder(images)  # [batch, 1, HIDDEN_SIZE]
        vision_tokens = self.linear_layer(vision_features)  # Map to GPT-2 input size

        # Append visual tokens at the start of the input sequence
        input_embeddings = self.text_decoder.transformer.wte(input_ids)
        embeddings = torch.cat([vision_tokens, input_embeddings], dim=1)

        outputs = self.text_decoder(inputs_embeds=embeddings, attention_mask=attention_mask)
        return outputs

# Load datasets
def load_data():
    cifar_trainloader, cifar_testloader = get_cifar10_loaders()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    eli5_dataset = ELI5Dataset(tokenizer, MAX_LENGTH=128, data_type='train')
    eli5_loader = DataLoader(eli5_dataset, batch_size=32, shuffle=True)
    
    visual_tuning_dataset = LLaVADataset(json_file='instruct_tuning/instruct.json',
                                        img_path='instruct_tuning/images',
                                        tokenizer=tokenizer, is_train=True)
    visual_loader = DataLoader(visual_tuning_dataset, batch_size=16, shuffle=True)
    return cifar_trainloader, eli5_loader, visual_loader, tokenizer

# Training function
def train(model, dataloader, optimizer, criterion):
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
        optimizer.step()

# Main Function
def main():
    # Initialize components
    cifar_trainloader, eli5_loader, visual_loader, tokenizer = load_data()
    vision_encoder = VisionEncoder().to(device)
    text_decoder = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model = VisionLanguageModel(vision_encoder, text_decoder).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Step 1: Train vision encoder on CIFAR-10
    print("Training Vision Encoder on CIFAR-10...")
    for epoch in range(NUM_EPOCHS):
        for images, _ in cifar_trainloader:
            images = images.to(device)
            optimizer.zero_grad()
            _ = vision_encoder(images)
            optimizer.step()

    # Step 2: Train text decoder
    print("Training Vision Encoder on CIFAR-10...")
    text_decoder.train()
    
    for batch in eli5_loader:
        input_ids = batch.to(device)  # Input IDs from ELI5 dataset
        labels = input_ids.clone()  # GPT-2 predicts next tokens
        
        optimizer.zero_grad()
        outputs = text_decoder(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    
    # Step 4: Fine-tune VLM on visual instruction tuning dataset
    print("Fine-tuning Vision-Language Model...")
    for epoch in range(NUM_EPOCHS):
        train(model, visual_loader, optimizer, criterion)

    # Save logits for test set
    test_loader = DataLoader(LLaVADataset('instruct_tuning/instruct.json', 'instruct_tuning/images', tokenizer, False),
                             batch_size=20, shuffle=False)
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)
            outputs = model(images, input_ids, attention_mask)
            all_logits.append(outputs.logits.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    np.save(LOGITS_FILE, all_logits)
    print(f"Logits saved to {LOGITS_FILE}")

if __name__ == "__main__":
    main()
