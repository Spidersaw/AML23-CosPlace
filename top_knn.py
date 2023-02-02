import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from model import network
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("query_image")
parser.add_argument("model_path")
parser.add_argument("dataset_path")
parser.add_argument("--k", default=5)
parser.add_argument("--backbone", default="resnet18")
parser.add_argument("--fc_output_dims", default=512)
args = parser.parse_args()
print(args)

model = network.GeoLocalizationNet(args.backbone, args.fc_output_dims)
model_dict = torch.load(args.model_path)

try:
    del model_dict["domain_discriminator.1.weight"]
    del model_dict["domain_discriminator.1.bias"]
    del model_dict["domain_discriminator.3.weight"]
    del model_dict["domain_discriminator.3.bias"]
    del model_dict["domain_discriminator.5.weight"]
    del model_dict["domain_discriminator.5.bias"]
except: 
    print("No GRL set")

model.load_state_dict(model_dict)

    
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# loading images in dataset folder 
my_dataset = ImageFolder(root=args.dataset_path, transform=data_transform)

def plot_knn(K):
    #open and reshape img according the imported ds 
    img = Image.open(args.query_image)
    img = data_transform(img).unsqueeze(0)
    embedding = model(img).detach().squeeze()

    # compute the distances between the target image and all the other images
    dists = []
    for i in range(len(my_dataset)):
        path, _ = my_dataset.imgs[i]
        if path == args.query_image:
            continue
        img = Image.open(path)
        img = data_transform(img).unsqueeze(0)
        emb = model(img).detach().squeeze()
        dist = torch.norm(embedding - emb, dim=0)
        dists.append((i, dist))

    # sort the distances in ascending order and get the indices of the top K nearest neighbors
    dists = sorted(dists, key=lambda x: x[1])
    top_k = [x[0] for x in dists[:K]]

    # plotting target and neighbors
    plt.figure(figsize=(15, 15))
    plt.subplot(K+1, 1, 1)
    img = Image.open(args.query_image)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Target image')
    for i, j in enumerate(top_k):
        plt.subplot(K+1, 1, i+2)
        img = Image.open(my_dataset.imgs[j][0])
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'{i+1}th nearest neighbor')
    plt.show()


#plot top K(argument) neighbors
plot_knn(args.k)