import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image


class Predictor():
    def __init__(self, dir, net, model_path='./model/model.pth', map_path="./data/map.csv"):
        self.directory = dir
        self.net = net
        self.model_path = model_path
        self.map_path = map_path 
        self.map = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path))
        self.net.eval()
        self.load_mapping()

    def load_mapping(self):
        self.map = pd.read_csv(self.map_path, header=None)

    def inference(self):
        # cwd = os.getcwd()
        # os.chdir(self.directory)
        filelist = os.listdir(self.directory)
        result = {}
        print('\n')
        print('************************************************************')
        print('*                   Drill Bit Classifier                   *')
        print('************************************************************')
        print('\n')
        print('Start Predicting ......\n')
        for file in filelist:
            image = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Resize((224, 224))
                ])(Image.open(os.path.join(self.directory, file)))
            image = image.reshape([1, image.size(dim=0), image.size(dim=1), image.size(dim=2)])
            outputs = self.net(image)
            _, predicted = torch.max(outputs.data, 1)
            size = self.map.iloc[predicted.item(),0]
            if size not in result.keys():
                result[size]=1
            else:
                result[size]+=1

            print(f'Predicted drill bit size for {file}: {size}')
        # os.chdir(cwd)
        print('\n------------ Summary --------------')
        print("{:<20} {:<20}".format('Bit Size', '# Predictions'))
        for k, v in result.items():
            print("{:<20} {:<20}".format(k, v))
            
