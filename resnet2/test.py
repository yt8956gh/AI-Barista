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
CUDA_DEVICES = 1
DATASET_ROOT2 = '/home/pwrai/0912test_photo_preprocessed/'
PATH_TO_WEIGHTS = './weights_pre_resnet101.pth'


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
    split = math.ceil(len(all_data_set)*0.01)  # extract 10% dataset as test-set
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(valid_idx)
    #print('test')
    #print(test_sampler)
        #train_set, test_set = torch.utils.data.random_split(train_set, [400, 115])
    print('train_set:',len(train_sampler),'test_set:',len(test_sampler))

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
    model.fc=nn.Linear(f(1600)*f(1066)*2048, 5)
    model.load_state_dict(torch.load(PATH_TO_WEIGHTS))

    #model=nn.DataParallel(model)f=lambda x:math.ceil(x/32-7+1)

    model = model.cuda(CUDA_DEVICES)
    model.eval()        #test

    # total_correct = 0
    # total = 0
    # class_correct = list(0. for i in enumerate(classes))
    # class_total = list(0. for i in enumerate(classes))

    with torch.no_grad():
        for inputs, labels in test_data_loader:
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))
            print('\n========== Labels ==========')
            # print(labels)
            # print([classes[label] for label in labels])
            outputs = model(inputs)
            tmp = F.softmax(outputs,dim=1)
            print("\n========== Softmax =========\n", tmp)
            _, predicted = torch.max(outputs.data, 1)       #預測機率最高的class
            print('\n========= Predicted ========\n')
            tmp = outputs.data.cpu().numpy()
            print(tmp.astype(int))

            '''
            # totoal
            # total += labels.size(0)
            # total_correct += (predicted == labels).sum().item()
            # c = (predicted == labels).squeeze()
            # batch size
            for i in range(labels.size(0)):
                label =labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            '''

    print('Accuracy on the ALL test images: %d %%'
          % (100 * total_correct / total))
    test_acc.append(total_correct/total)

    for i, c in enumerate(classes):
        print('Accuracy of %5s(%d photos) : %2d %%' % (
        c,class_total[i], 100 * class_correct[i] / class_total[i]))
    return(test_acc)


if __name__ == '__main__':
    test([], myDataloader())
