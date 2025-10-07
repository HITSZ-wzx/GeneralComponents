import argparse
import numpy as np
import time
import os
import torch
import re
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import heapq
import logging
from cifar_resnet import ResNet18, ResNet50, ResNet34
from utils import *
#from utils_CL import *
from PIL import Image

def train_step(model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels, is_poison) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

def test_step(model, criterion, data_loader, target):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    target_correct = 0
    target_num = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
            for i in range(len(pred)):
                a = pred[i].long()
                b = labels.data.view_as(pred)[i].long()
                if b == target:
                    if a == b:
                        target_correct += 1
                    target_num += 1
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    if target_num != 0:
        tar_acc = float(target_correct) / target_num
    else:
        tar_acc = 0
    return loss, acc, tar_acc

def attach_index(path, index, suffix=""):
    if re.search(suffix + "$", path):
        prefix, suffix = re.match(f"^(.*)({suffix})$", path).groups()
    else:
        prefix, suffix = path, ""
    return f"{prefix}_{index}{suffix}"

parser = argparse.ArgumentParser(description='Evaluate backdoor attack with different selection methods')
parser.add_argument('--model', default='resnet18', choices=['resnet18', 'resnet50', 'resnet34'])
parser.add_argument('--selection', default='res', choices=['random', 'loss', 'grad', 'forget', 'res', 'stealth'])
parser.add_argument('--res_sel', default='linear', choices=['max', 'exp', 'linear', 'log', 'square', 'num', 'third', 'poison'])
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=600, help='number of epochs to train (default: 200)')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--output_dir', type=str, default='/home/boot/STU/workspaces/wzx/Samples_select/save_metric_10_res', help='directory where to save metrics, same with the one used in cal_metric.py')
parser.add_argument('--result_dir', type=str, default='test_res', help='directory where to save results')
parser.add_argument('--model_dir', type=str, default='/home/boot/STU/workspaces/wzx/Samples_select/models/', help='directory where to save results')
parser.add_argument('--y_target', type=int, default=1)
parser.add_argument('--dataset', default='cifar10', help='dataset')
parser.add_argument('--num_levels', type=str, default="36:60:12")
parser.add_argument('--poison_rate', type=float, default=0.05)
parser.add_argument('--res_rate', type=float, default=1)
parser.add_argument('--backdoor_type', default='FTrojan', choices=['badnets', 'blend', 'quantize', 'narcissus', 'siba', 'combat','ground','clean','inputaware','FTrojan'])
parser.add_argument('--select_epoch', type=int, default=10, help='epoch which to calculate the stats')
parser.add_argument('--num_classes', type=int, default=10, help='num of the classes')
parser.add_argument('--blend_size', type=int, default=32, help='the size of blend image')
parser.add_argument('--data_dir', type=str, default='/home/amax/STU/DATASET/tiny-imagenet-200', help='directory where is dataset')
parser.add_argument('--type', type=str, default="0:0:0")
parser.add_argument('--save_trigger', type=str, default="/home/boot/STU/workspaces/wzx/Samples_select/save_triggerclean")
args = parser.parse_args()
use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if use_cuda else "cpu")
cudnn.benchmark = True
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
set_random_seed(args.seed)

if args.dataset == 'cifar10':
    train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='/home/boot/STU/workspaces/wzx/Samples_select/data', train=True, transform=transforms.ToTensor(), download=True)
    num_classes = 10
    test_dataset = datasets.CIFAR10(root='/home/boot/STU/workspaces/wzx/Samples_select/data', train=False, transform=transforms.ToTensor(), download=True)
elif args.dataset == 'cifar100':
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR100(root='./data100', train=True, transform=transforms.ToTensor(), download=True)
    num_classes = 100
    test_dataset = datasets.CIFAR100(root='./data100', train=False, transform=transforms.ToTensor(), download=True)
else :
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    num_classes = args.num_classes
    train_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=transforms.ToTensor())
    test_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'val'), transform=transforms.ToTensor())

if args.backdoor_type == 'badnets':
    checkboards = {}
    checkboards[0] = torch.Tensor([[0,0,1],[0,1,0],[1,0,1]]).repeat((3,1,1))
    checkboards[1] = torch.Tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).repeat((3, 1, 1))
    checkboards[2] = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).repeat((3, 1, 1))
    trigger = torch.zeros([3, 32, 32])
    if args.type != None:
        trigger_alpha = torch.zeros([3, 32, 32])
        trigger_alpha[:, 26:29, 17:20] = 1.0
    else:
        checkboard = torch.Tensor([[0,0,0],[0,0,0],[0,0,0]]).repeat((3,1,1))
        trigger[:, 26:29, 17:20] = checkboard
        trigger_alpha = torch.zeros([3, 32, 32])
        trigger_alpha[:, 26:29, 17:20] = 1.0
elif args.backdoor_type == 'blend':
    image_path = '/home/boot/STU/workspaces/wzx/bench/resource/blended/hello_kitty.jpeg'  # 替换为你的JPEG图片路径
    image = Image.open(image_path).convert('RGB')
    resized_image = image.resize((args.blend_size, args.blend_size), Image.ANTIALIAS)
    trigger_m = np.array(resized_image)
    trigger_m = torch.from_numpy(trigger_m)
    trigger_m = np.transpose(trigger_m, (2, 0, 1))
    trigger_m = trigger_m/255
    trigger_m = trigger_m.type(torch.FloatTensor)
    trigger= torch.zeros([3, 32, 32])
    trigger[:, 32 - args.blend_size:32, 32 - args.blend_size:32] = trigger_m
    trigger_alpha = torch.zeros([3, 32, 32])
    trigger_alpha[:, 32 - args.blend_size:32, 32 - args.blend_size:32] = 1.0
    if args.blend_size > 24:
        trigger_alpha *= 0.1
    elif args.blend_size > 16:
        trigger_alpha *= 0.4
    elif args.blend_size > 8:
        trigger_alpha *= 0.8
    else:
        trigger_alpha *= 1
elif args.backdoor_type == 'narcissus':
    save_path = '/home/boot/STU/workspaces/wzx/Narcissus/checkpoint/nar_' + args.dataset + '_' + str(args.y_target) +'.pt'
    #save_path = '/home/boot/STU/workspaces/wzx/remain/Narcissus-main/checkpoint/nnoise_2_1_3_01000.pth'
    if os.path.exists(save_path):
        temp_trigger = torch.load(save_path)
    trigger = temp_trigger.squeeze(0)

total_poison = int(len(train_dataset) * args.poison_rate)

if args.selection in ['loss', 'grad', 'forget', 'res', 'stealth']:
    stats_metric, stats_class, stats_inds = get_stats(args.selection, args.output_dir, args.select_epoch, args.seed, num_classes, args.y_target, args.res_sel, args.res_rate)
    metric_vals, metric_inds = [], []
    #只投毒了target-label数据
    for i in range(len(train_dataset)):
        if stats_class[i] == args.y_target:
            metric_vals.append(stats_metric[i])
            metric_inds.append(stats_inds[i])
    #largest_inds = heapq.nlargest(total_poison, range(len(metric_vals)), metric_vals.__getitem__)
    if args.selection != 'stealth':
        largest_inds = heapq.nlargest(total_poison, range(len(metric_vals)), metric_vals.__getitem__)
        poison_inds = [metric_inds[i] for i in largest_inds]
else:
    shuffle = np.random.permutation(len(train_dataset))
    k = 0
    poison_inds = []
    total_poison = len(train_dataset) * args.poison_rate
    for i in shuffle:
        if args.selection == 'poison':
            if train_dataset[i][1] != args.y_target and k < total_poison:
                poison_inds.append(i)
                k += 1
        else:
            if train_dataset[i][1] == args.y_target and k < total_poison:
                poison_inds.append(i)
                k += 1
#torch.save(poison_inds, "/home/boot/STU/workspaces/wzx/"+ args.res_sel + ".pt")
os.makedirs(args.result_dir, exist_ok=True)
logger = logging.getLogger()
if args.selection != 'stealth':
    args.poisoning_rate = len(poison_inds) *1.0/len(train_dataset)
    logger.info(str(len(poison_inds)))
logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.result_dir, 'output_{}.log'.format(args.seed))),
            logging.StreamHandler()
        ])
logger.info(args)
if args.model == 'resnet18':
    model = ResNet18(num_classes=num_classes)
elif args.model == 'resnet50':
    model = ResNet50(num_classes=num_classes)
elif args.model == 'resnet34':
    model = ResNet34(num_classes=num_classes)
elif args.model == 'AlexNet':
    from models.AlexNet import *
    model = AlexNet()
elif args.model == "SqueezeNet":
    from models.SqueezeNet import *
    model = SqueezeNet()
elif args.model == "VGG16":
    from models.VGG16 import *
    model = VGG16()
elif args.model == "GoogLeNet":
    from models.GoogLeNet import *
    model = GoogLeNet()
elif args.model == "DenseNet121":
    from models.DenseNet import *
    model = DenseNet121()
elif args.model == "MobileNet":
    from models.MobileNet import *
    model = MobileNet()
model = model.cuda()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = torch.nn.CrossEntropyLoss().to(device)
if args.backdoor_type == 'quantize':
    poison_train_set = Add_Clean_Label_Train_Trigger_Quantize(train_dataset, args.y_target, poison_inds, args.num_levels)
    poison_test_set = Add_Test_Trigger_Quantize(test_dataset, args.y_target, args.num_levels)
elif args.backdoor_type == 'narcissus':
    if args.selection == 'stealth':
        poison_train_set = Add_Clean_Label_Train_Trigger_NAR_stealth(train_dataset, trigger, args.y_target, metric_inds, total_poison)
    else :
        poison_train_set = Add_Clean_Label_Train_Trigger_NAR(train_dataset, trigger, args.y_target, poison_inds)
    poison_test_set = Add_Test_Trigger_NAR(test_dataset, trigger, args.y_target)
elif args.backdoor_type == 'siba':
    args.save_trigger = "/home/boot/STU/workspaces/wzx/remain/SIBA-main/save_trigger_" + str(num_classes) + "_" + str(args.y_target)
    poison_train_set = Add_Clean_Label_Train_Trigger_siba(train_dataset, args.y_target, poison_inds, args.save_trigger)
    poison_test_set = Add_Test_Trigger_siba(test_dataset, args.y_target, args.save_trigger)
elif args.backdoor_type == 'ground':
    poison_train_set = Add_Clean_Label_Train_Trigger_ground(train_dataset, args.y_target, poison_inds, args.dataset)
    poison_test_set = Add_Test_Trigger_ground(test_dataset, args.y_target, args.dataset)
elif args.backdoor_type == 'combat':
    poison_train_set = Add_Clean_Label_Train_Trigger_combat(train_dataset, args.y_target, poison_inds)
    poison_test_set = Add_Test_Trigger_combat(test_dataset, args.y_target)
elif args.backdoor_type == 'badnets':
    poison_train_set = Add_Clean_Label_Train_Trigger_badnets(train_dataset, trigger, args.y_target, trigger_alpha, poison_inds, args.type, checkboards)
    poison_test_set = Add_Test_Trigger_badnets(test_dataset, trigger, args.y_target, trigger_alpha, args.type, checkboards)
elif args.backdoor_type == 'inputaware':
    ckpt_path = "/home/boot/STU/workspaces/wzx/remain/input-aware-backdoor-attack-release-master/checkpoints96_"+str(args.y_target)+"_" + args.dataset + "/" + args.dataset + "/all2one"
    save_path = os.path.join(ckpt_path, "{}_{}_ckpt.pth.tar".format("all2one", args.dataset))
    #save_path = "/home/boot/STU/workspaces/wzx/BLearnDefense/pretrained/loaded/inputaware.pt"
    load = torch.load(save_path)
    model.load_state_dict(load["netC"], True)
    poison_test_set = load["poi_dataset"]
    poison_train_set = load["poi_dataset_train"]
    clean_train_set = load["clean_dataset_train"]
    labels_ = load["clean_labels"]
    model.eval()
    total_correct = 0
    total_loss = 0.0
    target_correct = 0
    target_num = 0
    dataset_ = list()
    with torch.no_grad():
        for images in poison_test_set:
            dataset_.append((images.to('cpu').numpy(),args.y_target))
            images = images.to(device)
            labels = torch.tensor([0]).to(device)
            output = model(images.unsqueeze(0))
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(poison_test_set)
    acc = float(total_correct) / len(poison_test_set)
    logger.info("test fininshed with acc %.4f", acc)
    path_to_save = "/home/boot/STU/workspaces/wzx/BLearnDefense/pretrained/" + args.dataset + "/tar"+str(args.y_target)+"/model_inputaware.pt"
    torch.save(model.state_dict(), path_to_save)
    total_correct = 0
    total_loss = 0.0
    target_correct = 0
    target_num = 0
    tdataset_ = list()
    with torch.no_grad():
        for images in poison_train_set:
            tdataset_.append((images.to('cpu').numpy(),args.y_target,1))
            images = images.to(device)
            labels = torch.tensor([0]).to(device)
            output = model(images.unsqueeze(0))
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    with torch.no_grad():
        for i in range(len(clean_train_set)):
            images = clean_train_set[i]
            label = labels_[i].cpu().item()
            tdataset_.append((images.to('cpu').numpy(),label,0))
            images = images.to(device)
            labels = torch.tensor([label]).to(device)
            output = model(images.unsqueeze(0))
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / (len(poison_train_set)+len(clean_train_set))
    acc = float(total_correct) / (len(poison_train_set)+len(clean_train_set))
    logger.info("train fininshed with acc %.4f", acc)
    res = {}
    res["train"] = tdataset_
    res["test"] = dataset_
    path_to_save = "/home/boot/STU/workspaces/wzx/BLearnDefense/pretrained/loaded/poiset_inputaware_"+ args.dataset + "_" + str(args.y_target)+".pt"
    logger.info("Saving finished")
    torch.save(res, path_to_save)
    #exit()
elif args.backdoor_type == 'FTrojan':
    if args.dataset == "cifar10":
        path = "/home/boot/STU/workspaces/wzx/Samples_select/resources/poi_dataset_FTrojan_" + "CIFAR10" + "_" + str(args.poison_rate) + "_tar" + str(args.y_target)+".pkl"
    else:
        path = "/home/boot/STU/workspaces/wzx/Samples_select/resources/poi_dataset_FTrojan_" + "CIFAR100" + "_" + str(args.poison_rate) + "_tar" + str(args.y_target)+".pkl"
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    poison_train_set = loaded_data["poi_trainset"]
    poison_test_set = loaded_data["poi_testset"]
else:
    if args.selection == 'stealth':
        poison_train_set = Add_Clean_Label_Train_Trigger_blend_stealth(train_dataset, trigger, args.y_target,
                                                               metric_inds, args.type, total_poison)
    else :
        poison_train_set = Add_Clean_Label_Train_Trigger(train_dataset, trigger, args.y_target, trigger_alpha, poison_inds, args.type)
    poison_test_set = Add_Test_Trigger(test_dataset, trigger, args.y_target, trigger_alpha, args.type)
poison_train_set = MyDataset(poison_train_set, train_transform)
if args.backdoor_type == 'combat':
    train_loader = DataLoader(poison_train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    trigger_loader = DataLoader(poison_test_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
else:
    train_loader = DataLoader(poison_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    trigger_loader = DataLoader(poison_test_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

if args.dataset == 'cifar10':
    model_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True,
                                      weight_decay=5e-4)
    scheduler = MultiStepLR(model_optimizer, milestones=[60, 90], gamma=0.1)
else:
    model_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=False,
                                      weight_decay=5e-4)
    scheduler = MultiStepLR(model_optimizer, milestones=[150, 225], gamma=0.1)

os.makedirs(args.result_dir, exist_ok=True)
logger = logging.getLogger()
if args.selection != 'stealth':
    args.poisoning_rate = len(poison_inds) *1.0/len(train_dataset)
    logger.info(str(len(poison_inds)))
logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.result_dir, 'output_{}.log'.format(args.seed))),
            logging.StreamHandler()
        ])
logger.info(args)
logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC \t TargetSum \t CleanSum')
clean_sum = 0.0
poison_sum = 0.0
max_cl = 0.70
max_po = 0
#save_path = "/home/boot/STU/workspaces/wzx/remain/parameter_backdoor-main/results816/checkpoint.pth"
'''save_path = "/home/boot/STU/workspaces/wzx/BLearnDefense/pretrained/"
save_path = os.path.join(os.path.join(os.path.join(save_path, args.dataset), "tar"+str(args.y_target)),"model_"+ args.backdoor_type+".pt")
save_path = "/home/boot/STU/workspaces/wzx/Samples_select/models/narcissus_linear_0.0005_1/0.9368_0.9156666666666666.pt"
a = torch.load(save_path)'''
#import pandas as pd  
  
# 读取CSV文件  
#df = pd.read_csv(save_path)
'''try:
    model.load_state_dict(torch.load(save_path), True)
except:
    try:
        model.load_state_dict(torch.load(save_path)['model'], True)
    except:
        model.load_state_dict(torch.load(save_path)['model_state_dict'], True)'''
for epoch in range(args.epochs):
    start = time.time()
    lr = model_optimizer.param_groups[0]['lr']
    train_loss, train_acc = train_step(model, criterion, model_optimizer, train_loader)
    cl_test_loss, cl_test_acc, cl_tar_acc= test_step(model, criterion, test_loader, args.y_target)
    po_test_loss, po_test_acc, po_tar_acc = test_step(model, criterion, trigger_loader, args.y_target)
    clean_sum = clean_sum + cl_test_acc
    poison_sum = poison_sum + po_test_acc
    po_tar_acc = 0.0
    cl_tar_acc = 0.0
    if (max_cl + max_po) < (cl_test_acc + po_test_acc):
        path = os.path.join(args.model_dir, args.result_dir)
        if not os.path.exists(path):
            os.mkdir(path)
        path_to_save = os.path.join(args.model_dir, args.result_dir, str(cl_test_acc)+"_"+str(po_test_acc)+".pt")
        torch.save(model.state_dict(), path_to_save)
        max_cl = cl_test_acc
        max_po = po_test_acc
    if epoch % 20 == 0:
        po_tar_acc = poison_sum / 20
        cl_tar_acc = clean_sum / 20
        clean_sum = 0.0
        poison_sum = 0.0
    scheduler.step()
    end = time.time()
    logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc, po_tar_acc, cl_tar_acc)


