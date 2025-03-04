from trainer import load_model
from data import load_data, create_data_loaders

import os
import torch

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE




os.environ["TORCH_HOME"] = "/ocean/projects/cis240109p/abollado/.cache"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)


train, test = load_data(full_dataset=False)
trainl, testl = create_data_loaders(train, test, batch_size=128)

model = load_model(name="mobilenet_v3", pretrained_path="models/final_models/mobilenet_v3_not_freeze/model.pt", nofreeze=False)
model.to(device)
model.eval()

layers  = [model.features, model.avgpool]
layers2 = [model.classifier[0], model.classifier[1], model.classifier[2]]
layers3 = [model.classifier[3][0], model.classifier[3][1], model.classifier[3][2]]

print("✅ Model loaded successfully!")



embeddings = []
labels_emb = []

for x, labels in trainl:
    x, labels = x.to(device), labels.float().to(device)
    labels = labels.view(-1, 1)
    # x = model(x)
    for layer in layers:
        x = layer(x)
    x = x.reshape(-1, 960)
    for layer in layers2:
        x = layer(x)
    for layer in layers3:
        x = layer(x)
    embeddings.append(x)      # at this point x is a tensor of shape (batch_size, 1000) -> before the last fc layer: 1000 -> 1
    labels_emb.append(labels)
    


X = torch.concatenate(embeddings, dim=0).cpu().detach().numpy()
y = torch.concatenate(labels_emb, dim=0).cpu().detach().numpy().reshape(-1)

tsne = TSNE(n_components=2, random_state=42)

X2d = tsne.fit_transform(X)

# Create a DataFrame for seaborn
df = pd.DataFrame({
    'Component 1': X2d[:, 0],
    'Component 2': X2d[:, 1],
    'Label': y.astype(int)  # Ensure labels are integers (0 or 1)
})

# Plot using seaborn's scatterplot with hue for binary labels
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='Component 1', y='Component 2', hue='Label',
                palette=['blue', 'orange'], s=10, alpha=0.7)
plt.title('t-SNE Visualization of 1000-dimensional Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Label')
plt.savefig('t-SNE.png')