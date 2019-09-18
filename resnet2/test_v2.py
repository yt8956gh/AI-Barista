import torch
#from utils import parse_args
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torchvision.models as models
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from dataset import IMAGE_Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import math
import os
CUDA_DEVICES = 0
DATASET_ROOT2 = '/home/pwrai/0912test_photo_preprocessed/'
PATH_TO_WEIGHTS = './models/Model_ResNet-Reg_5-level.pth'

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"   # "%d" % CUDA_DEVICES


def myDataloader():
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    #print(DATASET_ROOT)
    all_data_set = IMAGE_Dataset(Path(DATASET_ROOT2), data_transform)
    
    #print('set:',len(train_set))
    indices = list(range(len(all_data_set)))
    #print('old',indices)
    np.random.seed(1)
    np.random.shuffle(indices)
    #print('new',indices)
    split = math.ceil(len(all_data_set)*1)  # extract 10% dataset as test-set
    valid_idx = indices[:split]
    test_sampler = SubsetRandomSampler(valid_idx)
    #print('test')
    #print(test_sampler)
    #train_set, test_set = torch.utils.data.random_split(train_set, [400, 115])
    print('test_set: ',len(test_sampler))

    test_data_loader=DataLoader(
                dataset=all_data_set,
                batch_size=8,
                shuffle=False,
                num_workers=0,
                sampler=test_sampler)
    
    return test_data_loader


def test(test_acc,test_data_loader):

    '''
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    test_set = IMAGE_Dataset(Path(DATASET_ROOT2), data_transform)
    data_loader = DataLoader(
        dataset=test_set, batch_size=32, shuffle=True, num_workers=1)
    '''

    classes = [_dir.name for _dir in Path(DATASET_ROOT2).glob('*')]     #資料夾名稱

    model=models.resnet101()
    f=lambda x:math.ceil(x/32-7+1)
    
    # model.load_state_dict(torch.load(PATH_TO_WEIGHTS))
    model = torch.load(PATH_TO_WEIGHTS)
    model.fc=nn.Linear(f(1600)*f(1066)*2048, 5)


    #model=nn.DataParallel(model)

    model = model.cuda(CUDA_DEVICES)
    model.eval()        #test
    
    '''
    total_correct = 0
    total = 0
    class_correct = list(0. for i in enumerate(classes))
    class_total = list(0. for i in enumerate(classes))
    '''

    criterion = nn.L1Loss()
    criterion2 = nn.MSELoss()

    test_loss = []
    test_loss2 = []

    with torch.no_grad():

        for inputs, labels in test_data_loader:
            testing_loss = 0.0
            testing_loss2 = 0.0
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))
            print('\n==================== Labels ====================\n')
            print(labels)
            # print([classes[label] for label in labels])
            outputs = model(inputs)
            tmp = F.softmax(outputs,dim=1)
            tmp *= 100

            print("\n================== Softmax =====================\n", tmp.int())
            # _, predicted = torch.max(outputs.data, 1)       #預測機率最高的class
            # _ , preds = torch.max(outputs.data, 1)
            outputs = outputs.squeeze(1)
            tmp = outputs.data.cpu().numpy()
            print("\n================= Predictions ==================\n")
            print(tmp.astype(int))

            loss = criterion(outputs, labels)
            loss2 = criterion2(outputs, labels)
            print("\n================= Batch Loss ===================\n")
            print(f"Testing : {loss.data:.2f}")
            print(f"MSELoss : {loss2.data:.2f}\n")
            # print(f"loss in epoch: {train_loss:.2f}")
            # print(f"MSEloss in epoch: {train_loss2:.2f}\n")

            testing_loss += loss.item() * inputs.size(0)
            testing_loss2 += loss2.item() * inputs.size(0)

        # Calulate Loss and MSELoss in current epoch       
        testing_loss = testing_loss / len(test_data_loader)
        testing_loss2 = testing_loss2 / len(test_data_loader)
        # train_acc.append(training_acc)        #save each 10 epochs accuracy
        print("\n================= All Loss ===================\n")
        print(f'Training:', testing_loss) 
        print(f'MSEloss :', testing_loss2) 


if __name__ == '__main__':

    dataset = myDataloader()
    test([], dataset)
