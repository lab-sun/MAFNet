# By Zhen FENG, Aug. 5, 2022
# Email: zfeng@outlook.com

import os, argparse, time, datetime, stat, shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from util.MF_dataset import MF_dataset
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from util.util import getScores
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from model import MAFNet

#############################################################################################
parser = argparse.ArgumentParser(description='Train with pytorch')
############################################################################################# 
parser.add_argument('--model_name', '-m', type=str, default='MAFNet')
parser.add_argument('--batch_size', '-b', type=int, default=4) 
parser.add_argument('--lr_start', '-ls', type=float, default=0.1)
parser.add_argument('--gpu', '-g', type=int, default=0)
parser.add_argument('--img_height', '-ih', type=int, default=512) 
parser.add_argument('--img_width', '-iw', type=int, default=512)  
#############################################################################################
parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
parser.add_argument('--epoch_max', '-em', type=int, default=200) # please stop training mannully 
parser.add_argument('--epoch_from', '-ef', type=int, default=0) 
parser.add_argument('--num_workers', '-j', type=int, default=8)
parser.add_argument('--n_class', '-nc', type=int, default=2)
parser.add_argument('--data_dir', '-dr', type=str, default='./dataset/')
args = parser.parse_args()
#############################################################################################

augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0),
    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.15, prob=0.9),
    # RandomNoise(noise_range=5, prob=0.9),
]

def train(epo, model, train_loader, optimizer):
    model.train()
    for it, (rgb, tdisp,labels, names) in enumerate(train_loader):
        rgb = Variable(rgb).cuda(args.gpu)
        tdisp = Variable(tdisp).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        start_t = time.time() # time.time() returns the current time
        optimizer.zero_grad()
        logits = model(rgb,tdisp)
        loss = F.cross_entropy(logits, labels)  # Note that the cross_entropy function has already include the softmax function
        loss.backward()
        optimizer.step()
        lr_this_epo=0
        for param_group in optimizer.param_groups:
            lr_this_epo = param_group['lr']
        print('Train: %s, epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f, time %s' \
            % (args.model_name, epo, args.epoch_max, it+1, len(train_loader), lr_this_epo, len(names)/(time.time()-start_t), float(loss),
              datetime.datetime.now().replace(microsecond=0)-start_datetime))
        if accIter['train'] % 1 == 0:
            writer.add_scalar('Train/loss', loss, accIter['train'])
        view_figure = True # note that I have not colorized the GT and predictions here
        if accIter['train'] % 50 == 0:
            if view_figure:
                input_rgb_images = vutils.make_grid(rgb, nrow=8, padding=10) 
                writer.add_image('Train/input_rgb_images', input_rgb_images, accIter['train'])
                scale = max(1, 255//args.n_class) # label (0,1,2..) is invisable, multiply a constant for visualization
                groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*512*512 -> mini_batch*1*512*512
                groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)  # change to 3-channel for visualization
                groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                writer.add_image('Train/groudtruth_images', groudtruth_images, accIter['train'])
                predicted_tensor = logits.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*512*512 -> mini_batch*512*512 -> mini_batch*1*512*512
                predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor),1) # change to 3-channel for visualization, mini_batch*1*512*512
                predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                writer.add_image('Train/predicted_images', predicted_images, accIter['train'])
        accIter['train'] = accIter['train'] + 1

def validation(epo, model, val_loader): 
    model.eval()
    with torch.no_grad():
        for it, (rgb,tdisp, labels, names) in enumerate(val_loader):
            rgb = Variable(rgb).cuda(args.gpu)
            tdisp = Variable(tdisp).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            start_t = time.time() # time.time() returns the current time
            logits = model(rgb,tdisp)
            loss = F.cross_entropy(logits, labels)  # Note that the cross_entropy function has already include the softmax function
            print('Val: %s, epo %s/%s, iter %s/%s, %.2f img/sec, loss %.4f, time %s' \
                  % (args.model_name, epo, args.epoch_max, it + 1, len(val_loader), len(names)/(time.time()-start_t), float(loss),
                    datetime.datetime.now().replace(microsecond=0)-start_datetime))
            if accIter['val'] % 1 == 0:
                writer.add_scalar('Validation/loss', loss, accIter['val'])
            view_figure = False  # note that I have not colorized the GT and predictions here
            if accIter['val'] % 100 == 0:
                if view_figure:
                    input_rgb_images = vutils.make_grid(rgb, nrow=8, padding=10)  
                    writer.add_image('Validation/input_rgb_images', input_rgb_images, accIter['val'])
                    scale = max(1, 255 // args.n_class)  # label (0,1,2..) is invisable, multiply a constant for visualization
                    groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*512*512 -> mini_batch*1*512*512
                    groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)  # change to 3-channel for visualization
                    groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/groudtruth_images', groudtruth_images, accIter['val'])
                    predicted_tensor = logits.argmax(1).unsqueeze(1)*scale  # mini_batch*args.n_class*512*512 -> mini_batch*512*512 -> mini_batch*1*512*512
                    predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor), 1)  # change to 3-channel for visualization, mini_batch*1*512*512
                    predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/predicted_images', predicted_images, accIter['val'])
            accIter['val'] += 1

def testing(epo, model, test_loader):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["unlabeled", "pothole"]
    testing_results_file = os.path.join(weight_dir, 'testing_results_file.txt')
    with torch.no_grad():
        for it, (rgb, tdisp, labels,names) in enumerate(test_loader):
            rgb = Variable(rgb).cuda(args.gpu)
            tdisp = Variable(tdisp).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            logits = model(rgb,tdisp)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1]) # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            print('Test: %s, epo %s/%s, iter %s/%s, time %s' % (args.model_name, epo, args.epoch_max, it+1, len(test_loader),
                 datetime.datetime.now().replace(microsecond=0)-start_datetime))
    
    globalacc, pre_p, recall_p, Fsc_p, iou_p, pre_b, recall_b, Fsc_b, iou_b = getScores(conf_total)
    print ('Epoch {0:} glob acc : {1:.4f}, pre_p : {2:.4f}, recall_p : {3:.4f}, F_score_p : {4:.4f}, IoU_p : {5:.4f}'.format('pothole', globalacc*100, pre_p*100, recall_p*100, Fsc_p*100, iou_p*100))
    print ('Epoch {0:} glob acc : {1:.4f}, pre_b : {2:.4f}, recall_b : {3:.4f}, F_score_b : {4:.4f}, IoU_b : {5:.4f}'.format('background', globalacc*100, pre_b*100, recall_b*100, Fsc_b*100, iou_b*100))
    print ('Average pre :{0:.4f}, recall: {1:.4f}, F_score: {2:.4f}, IoU : {3:.4f}'.format( (pre_b+pre_p)/2*100, (recall_b+recall_p)/2*100, (Fsc_b+Fsc_p)/2*100, (iou_b+iou_p)/2*100))

    precision = [pre_b,pre_p]
    recall = [recall_b,recall_p]
    Fsc = [Fsc_b,Fsc_p]
    IoU = [iou_b,iou_p]
    
    writer.add_scalar('Test/average_precision', np.mean(np.nan_to_num(precision)), epo)
    writer.add_scalar('Test/average_recall', np.mean(np.nan_to_num(recall)), epo)
    writer.add_scalar('Test/average_Fsc', np.mean(np.nan_to_num(Fsc)), epo)
    writer.add_scalar('Test/average_IoU', np.mean(np.nan_to_num(IoU)), epo)
    for i in range(len(precision)):
        writer.add_scalar("Test(class)/precision_class_%s" % label_list[i], precision[i], epo)
        writer.add_scalar("Test(class)/recall_class_%s"% label_list[i], recall[i],epo)
        writer.add_scalar("Test(class)/Fsc_class_%s"% label_list[i], Fsc[i],epo)
        writer.add_scalar('Test(class)/Iou_class_%s'% label_list[i], IoU[i], epo)

    if epo==0:
        with open(testing_results_file, 'w') as f:
            f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" %(args.model_name, args.lr_start, args.batch_size, datetime.date.today()))
            f.write("# epoch: unlabeled, pothole, average(nan_to_num). (Pre %, Acc %, Fsc %, IoU %)\n")
    with open(testing_results_file, 'a') as f:
        f.write(str(epo)+': ')
        for i in range(len(precision)):
            f.write('%0.4f, %0.4f, %0.4f, %0.4f ' % (100*precision[i], 100*recall[i], 100*Fsc[i], 100*IoU[i]))
        f.write('%0.4f, %0.4f, %0.4f, %0.4f\n' % (100*np.mean(np.nan_to_num(precision)), 100*np.mean(np.nan_to_num(recall)), 100*np.mean(np.nan_to_num(Fsc)), 100*np.mean(np.nan_to_num(IoU))))
    print('saving testing results.')
    with open(testing_results_file, "r") as file:
        writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo)

if __name__ == '__main__':
   
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    model = eval(args.model_name)(n_class=args.n_class, input_h=args.img_height, input_w=args.img_width)
    if args.gpu >= 0: model.cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)

    # preparing folders
    if os.path.exists("./runs"):
        shutil.rmtree("./runs")
    weight_dir = os.path.join("./runs", args.model_name)
    os.makedirs(weight_dir)
    os.chmod(weight_dir, stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
 
    writer = SummaryWriter("./runs/tensorboard_log")
    os.chmod("./runs/tensorboard_log", stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
    os.chmod("./runs", stat.S_IRWXO) 

    print('training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)


    preweight = False

    if preweight ==True:
        pretrained_weight = torch.load("./weights_backup/MAFNet/final.pth", map_location = lambda storage, loc: storage.cuda(args.gpu))
        own_state = model.state_dict()
        for name, param in pretrained_weight.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)  
            print(name)
        print('done!')

    train_dataset = MF_dataset(data_dir=args.data_dir, split='train', transform=augmentation_methods)
    val_dataset  = MF_dataset(data_dir=args.data_dir, split='validation')
    test_dataset = MF_dataset(data_dir=args.data_dir, split='test')

    train_loader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    val_loader  = DataLoader(
        dataset     = val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    test_loader = DataLoader(
        dataset      = test_dataset,
        batch_size   = args.batch_size,
        shuffle      = False,
        num_workers  = args.num_workers,
        pin_memory   = True,
        drop_last    = False
    )
    start_datetime = datetime.datetime.now().replace(microsecond=0)
    accIter = {'train': 0, 'val': 0}
    for epo in range(args.epoch_from, args.epoch_max):
        print('\ntrain %s, epo #%s begin...' % (args.model_name, epo))
        #scheduler.step() # if using pytorch 0.4.1, please put this statement here 
        train(epo, model, train_loader, optimizer)
        validation(epo, model, val_loader)

        checkpoint_model_file = os.path.join(weight_dir, str(epo) + '.pth')
        print('saving check point %s: ' % checkpoint_model_file)
        torch.save(model.state_dict(), checkpoint_model_file)

        testing(epo, model, test_loader)
        scheduler.step() # if using pytorch 1.1 or above, please put this statement here
