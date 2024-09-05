# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Lumbar_Model(nn.Module):
  def __init__(self,input_shape:int,hidden_units:int,output_shape:int):
    super().__init__()
    self.conv_block_1=nn.Sequential(
        nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25)
    )
    self.conv_block_2=nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units*2,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units*2),
        nn.Conv2d(in_channels=hidden_units*2,out_channels=hidden_units*2,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units*2),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25)
    )
    self.conv_block_3=nn.Sequential(
        nn.Conv2d(in_channels=hidden_units*2,out_channels=hidden_units*4,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units*4),
        nn.Conv2d(in_channels=hidden_units*4,out_channels=hidden_units*4,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units*4),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25)
    )
    self.classifier=nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*8*8,out_features=output_shape)
    )
  def forward(self,x:torch.Tensor):
    x=self.conv_block_1(x)
    x=self.conv_block_2(x)
    x=self.conv_block_3(x)
    x=self.classifier(x)
    return x

Lumber_list=['processed_lsd', 'processed_osf', 'processed_spider', 'processed_tseg']

class Pneumonia_Model(nn.Module):
  def __init__(self,input_shape:int,hidden_units:int,output_shape:int):
    super().__init__()
    self.conv_block_1=nn.Sequential(
        nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25)
    )
    self.conv_block_2=nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units*2,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units*2),
        nn.Conv2d(in_channels=hidden_units*2,out_channels=hidden_units*2,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units*2),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25)
    )
    self.conv_block_3=nn.Sequential(
        nn.Conv2d(in_channels=hidden_units*2,out_channels=hidden_units*4,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units*4),
        nn.Conv2d(in_channels=hidden_units*4,out_channels=hidden_units*4,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units*4),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25)
    )
    self.classifier=nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*8*8,out_features=output_shape)
    )
  def forward(self,x:torch.Tensor):
    x=self.conv_block_1(x)
    x=self.conv_block_2(x)
    x=self.conv_block_3(x)
    x=self.classifier(x)
    return x
  
Pneumonia_list=['NORMAL', 'PNEUMONIA']


class Covid19_Model(nn.Module):
  def __init__(self,input_shape:int,hidden_units:int,output_shape:int):
    super().__init__()
    self.conv_block_1=nn.Sequential(
        nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25)
    )
    self.conv_block_2=nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units),
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25)
    )
    self.classifier=nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*8*8,out_features=output_shape)
    )
  def forward(self,x:torch.Tensor):
    x=self.conv_block_1(x)
    x=self.conv_block_2(x)
    
    x=self.classifier(x)
    return x
  
Covid19_list=['Covid', 'Normal', 'Viral Pneumonia']

class Brain_MRI_Model(nn.Module):
  def __init__(self,input_shape:int,hidden_units:int,output_shape:int):
    super().__init__()
    self.conv_block_1=nn.Sequential(
        nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25)
    )
    self.conv_block_2=nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units*2,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units*2),
        nn.Conv2d(in_channels=hidden_units*2,out_channels=hidden_units*2,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(hidden_units*2),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.25)
    )

    self.classifier=nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*8*8*2,out_features=output_shape)
    )
  def forward(self,x:torch.Tensor):
    x=self.conv_block_1(x)
    x=self.conv_block_2(x)
    x=self.classifier(x)
    return x
  
Brain_MRI_list=['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
# Initialize the model and load the state dict

torch.manual_seed(42)

lumber_model=Lumbar_Model(input_shape=1,hidden_units=32,output_shape=4).to(device)
lumber_model.load_state_dict(torch.load(f="Lumbar.pth",map_location=device))
lumber_model.eval() 

pneumonia_model=Pneumonia_Model(input_shape=1,hidden_units=32,output_shape=2).to(device)
pneumonia_model.load_state_dict(torch.load(f="Pneumonia.pth",map_location=device))
pneumonia_model.eval() 

covid19_model=Covid19_Model(input_shape=1,hidden_units=32,output_shape=3).to(device)
covid19_model.load_state_dict(torch.load(f="Covid_19.pth",map_location=device))
covid19_model.eval() 

brain_mri_model=Brain_MRI_Model(input_shape=1,hidden_units=32,output_shape=4).to(device)
brain_mri_model.load_state_dict(torch.load(f="Brain_MRI.pth",map_location=device))
brain_mri_model.eval() 

# Preprocess function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Adjust based on your model's input size
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize([0.5], [0.5]) 
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

# Prediction route
@app.post("/predict_lumbar/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = lumber_model(image_tensor)
        _, predicted = torch.max(output, 1)
    return JSONResponse(content={"prediction": Lumber_list[predicted.item()]})

@app.post("/predict_pneumonia/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = pneumonia_model(image_tensor)
        _, predicted = torch.max(output, 1)
    return JSONResponse(content={"prediction": Pneumonia_list[predicted.item()]})

@app.post("/predict_covid19/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = covid19_model(image_tensor)
        _, predicted = torch.max(output, 1)
    return JSONResponse(content={"prediction": Covid19_list[predicted.item()]})

@app.post("/predict_brain_mri/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = brain_mri_model(image_tensor)
        _, predicted = torch.max(output, 1)
    return JSONResponse(content={"prediction": Brain_MRI_list[predicted.item()]})


