# imports
import argparse
import numpy as np
import torch
import json
from torch import device, nn
from PIL import Image
from torchvision import transforms, models, datasets
from matplotlib import pyplot as plt

import torch.nn.functional as F

class Network(nn.Module):
    '''feed-forward network as a classifier, using ReLU activations and dropout '''
    def __init__(self, input_units, h1_units, h2_units, output_units):
        super().__init__()
        # Defining the layers
        self.fc1 = nn.Linear(input_units, h1_units)        
        # Inputs to hidden layer linear transformation
        self.fc2 = nn.Linear(h1_units, h2_units)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(h2_units, output_units)
        
    def forward(self, x):       
        ''' Forward pass through the network, returns the output logits '''
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.output(x)
        x = F.softmax(x, dim=1)
        
        return x

def load_checkpoint(model, checkpoint_file, weights_only=True):
    #model = torch.load(checkpoint_file, weights_only=True)
    #model = Network()
    model.load_state_dict(torch.load(checkpoint_file))#, weights_only))
    print(model)
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
    # Process a PIL image for use in a PyTorch model    
    transform = transforms.Compose([transforms.PILToTensor()])
    tensor = transform(image)
    
    return tensor

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    plt.show()
    
    return ax

def predict(image, model, test_data, classes, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    #classes = [ "a", "b", "c", "d", "e"]
    
    model.eval()
    
    x = test_data[0][0] # Image tensor
    y = test_data[0][1] # Label (class index)
    
    # Calculate the class probabilities (softmax) for img
    ps = torch.exp(model(image))
    
    with torch.no_grad():
        x = x.unsqueeze(0)  # Now shape will be [1, 3, 224, 224]

        x = x.to(device)
        prediction = model(x)
        
        predicted = classes[prediction[0].argmax(0)]
        actual = classes[y]
        
        print(f"Predicted {predicted}, Actual: {actual}")
    
    return ps, classes

def test_data(model, test_transforms, sample_image):
    '''Performs validation on the test set'''

    # label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # apply the transformation to the image then print the new shape
    transformed_image = test_transforms(sample_image)
    transformed_image.shape

    # prepare a batch for the network
    # unsqueeze(input, dim) turns an n.d. tensor into an (n+1).d ternsor
    batch = torch.unsqueeze(transformed_image, 0)

    # sets the PyTorch model to evaluation mode
    model.eval()

    # output batch to VGG
    out = model(batch)
    out.shape

    # classification stage
    # set torch to sour item desceningly
    _, indices = torch.sort(out, descending=True)
    #classify the image
    # Assuming indices is a tensor of indices
    top_indices = indices[0][:10].tolist()  # Convert to a list of integers
    print(top_indices)

    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    #[(cat_to_name[idx], percentage[idx].item()) for idx in indices[0][:10]]
    # Create the percentage list
    percentage_list = [(cat_to_name[idx.item()], percentage[idx].item()) for idx in top_indices]
    print(percentage_list)

def parse_arguments():
    None# Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # parse the different types of arguments
    parser.add_argument('--input', type = str, default = 'flowers/test/1/image_06743.jpg', help= 'the filepath and filename of image')
    parser.add_argument('--checkpoint', type = str, default = 'flower_classification_model.pth', help= 'filepath and filenane of checkpoint file')
    parser.add_argument('--top_k', type = int, default = 3, help= 'sets number of top K most likely classes')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'json file used for category mapping')
    parser.add_argument('--gpu', default = False, help= 'Whether to use GPU for inference' )

    # Return parsed argument collection
    return parser.parse_args()

# START OF PROGRAM #
if __name__ == "__main__":

    command_args = parse_arguments()

    # print command line arguments
    print('Image file:', command_args.input)
    print('Checkpoint:', command_args.checkpoint)
    print('Top K classes:', command_args.top_k)
    print('Category names file:', command_args.category_names)
    print('GPU for inference:', command_args.gpu)
    
    model = models.vgg11(pretrained=True)
    #print(model)
    checkpoint = load_checkpoint(model, command_args.checkpoint)

    # data setup
    test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    test_set = datasets.ImageFolder('flowers/test' , transform= test_transforms)
   

    # load and resize the image
    image = Image.open(command_args.input)
    image.resize((224,224)) #resizes image to 224x24 pixels
    
    #test_data(model, test_transforms, image)

    # map labes then make prediction
    with open(command_args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    # Get tensor from image
    tensor = process_image(image)
    #type(tensor)
    predict(image, model, test_set, cat_to_name, command_args.top_k)

    #Display an image along with the top 5 classes
    
    #type(tensor)
    #ax = imshow(tensor)
    #flower.show()

    image.close()