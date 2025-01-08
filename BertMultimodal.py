import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from transformers import get_scheduler
import torch.nn as nn
from transformers import BertModel
from transformers import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

df = pd.read_csv('./Music Dataset Lyrics and Metadata from 1950 to 2019/tcc_ceds_music.csv')  

lyrics = df['lyrics']
audio_features = df[['loudness', 'acousticness', 'danceability', 'instrumentalness']]
topics = df['topic']
#norm audio
scaler = MinMaxScaler()
audio_features = scaler.fit_transform(audio_features)
#labels to numbers
label_encoder = LabelEncoder()
encoded_topics = label_encoder.fit_transform(topics)

# split
train_lyrics, test_lyrics, train_audio, test_audio, train_topics, test_topics = train_test_split(
    lyrics, audio_features, encoded_topics, test_size=0.2, random_state=42
)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_texts(texts, tokenizer, max_len=512):
    return tokenizer(
        texts.tolist(),
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )


train_encodings = tokenize_texts(train_lyrics, tokenizer)
test_encodings = tokenize_texts(test_lyrics, tokenizer)

# tensorize
train_labels = torch.tensor(train_topics, dtype=torch.long)
test_labels = torch.tensor(test_topics, dtype=torch.long)

# moar tensorize
train_audio = torch.tensor(train_audio, dtype=torch.float32)
test_audio = torch.tensor(test_audio, dtype=torch.float32)

class MultimodalDataset(Dataset):
    def __init__(self, encodings, audio_features, labels):
        self.encodings = encodings
        self.audio_features = audio_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['audio_features'] = self.audio_features[idx]
        item['labels'] = self.labels[idx]
        return item

# Datasets
train_dataset = MultimodalDataset(train_encodings, train_audio, train_labels)
test_dataset = MultimodalDataset(test_encodings, test_audio, test_labels)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

class MultimodalBERTClassifier(nn.Module):
    def __init__(self, bert_model, audio_input_dim, num_classes):
        super(MultimodalBERTClassifier, self).__init__()
        self.bert = bert_model
        self.audio_fc = nn.Sequential(
            nn.Linear(audio_input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, audio_features):
        # berting the text
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.pooler_output  # (batch_size, 768)

        audio_features = self.audio_fc(audio_features)  # (batch_size, 128)
        audio_features = audio_features * 10

        # Concatenating
        combined_features = torch.cat((text_features * 0.7, audio_features * 0.3), dim=1)

        # classifying
        logits = self.classifier(combined_features)
        return logits
#load pretrained bert
bert_model = BertModel.from_pretrained('bert-base-uncased')
num_classes = len(label_encoder.classes_)
audio_input_dim = train_audio.shape[1]
#instantiating model
model = MultimodalBERTClassifier(bert_model, audio_input_dim, num_classes)
model.to(device)

# use adam and cel 
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = CrossEntropyLoss()
#scheduler for stability
epochs = 3
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=len(train_loader) * epochs)

for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        audio_features = batch['audio_features'].to(device)
        labels = batch['labels'].to(device)

        # palante
        logits = model(input_ids, attention_mask, audio_features)
        loss = criterion(logits, labels)

        # patras
        optimizer.zero_grad()
        loss.backward()
        
        #clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        scheduler.step()

        # nice bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
#eval time!
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        audio_features = batch['audio_features'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask, audio_features)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# metrics
accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {accuracy}")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
