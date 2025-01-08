import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# load things
df = pd.read_csv('./Music Dataset Lyrics and Metadata from 1950 to 2019/tcc_ceds_music.csv')

lyrics = df['lyrics']
topics = df['topic']

label_encoder = LabelEncoder()
encoded_topics = label_encoder.fit_transform(topics)
# split things
train_lyrics, test_lyrics, train_topics, test_topics = train_test_split(
    lyrics, encoded_topics, test_size=0.2, random_state=42
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

train_labels = torch.tensor(train_topics, dtype=torch.long)
test_labels = torch.tensor(test_topics, dtype=torch.long)

# Dataset class for lyyyrics
class LyricsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

train_dataset = LyricsDataset(train_encodings, train_labels)
test_dataset = LyricsDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# load pretrained 
num_classes = len(label_encoder.classes_)  # num topics
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
model.to(device)

# use Adam 
optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 3

for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        # to gpu
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)

        # loss 
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        # patras
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # noice bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

# Eval time !
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)

        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# metrics
accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {accuracy}")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
