# By Zhen FENG, Aug. 5, 2022
# Email: zfeng94@outlook.com

import os, argparse, time, datetime, sys, shutil, stat, torch
from torchsummary import summary
import numpy as np 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.MF_dataset import MF_dataset 
from util.util import visualize,getScores
from sklearn.metrics import confusion_matrix
from scipy.io import savemat 
from model import MAFNet

#############################################################################################
parser = argparse.ArgumentParser(description='Test with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='MAFNet')
parser.add_argument('--weight_name', '-w', type=str, default='MAFNet') 
parser.add_argument('--file_name', '-f', type=str, default='final.pth')
parser.add_argument('--dataset_split', '-d', type=str, default='test')
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--img_height', '-ih', type=int, default=512) 
parser.add_argument('--img_width', '-iw', type=int, default=512)  
parser.add_argument('--num_workers', '-j', type=int, default=16)
parser.add_argument('--n_class', '-nc', type=int, default=2)
parser.add_argument('--data_dir', '-dr', type=str, default='./dataset/')
parser.add_argument('--model_dir', '-wd', type=str, default='./weights_backup/')
args = parser.parse_args()
#############################################################################################
 
if __name__ == '__main__':
  
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    # prepare save direcotry
    if os.path.exists("./runs"):
        print("previous \"./runs\" folder exist, will delete this folder")
        shutil.rmtree("./runs")
    os.makedirs("./runs")
    os.chmod("./runs", stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
    model_dir = os.path.join(args.model_dir, args.weight_name)
    if os.path.exists(model_dir) is False:
        sys.exit("the %s does not exit." %(model_dir))
    model_file = os.path.join(model_dir, args.file_name)
    print(model_file)
    if os.path.exists(model_file) is True:
        print('use the final model file.')
    else:
        sys.exit('no model file found.') 
    print('testing %s: %s on GPU #%d with pytorch' % (args.model_name, args.weight_name, args.gpu))
    
    conf_total = np.zeros((args.n_class, args.n_class))

    model = eval(args.model_name)(n_class=args.n_class, input_h=args.img_height, input_w=args.img_width)

    if args.gpu >= 0: model.cuda(args.gpu)
    print('loading model file %s... ' % model_file)
    pretrained_weight = torch.load(model_file, map_location = lambda storage, loc: storage.cuda(args.gpu))
    own_state = model.state_dict()
    for name, param in pretrained_weight.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)  
    print('done!')

    batch_size = 1 # do not change this parameter!	
    test_dataset  = MF_dataset(data_dir=args.data_dir, split=args.dataset_split, input_h=args.img_height, input_w=args.img_width)
    test_loader  = DataLoader(
        dataset     = test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    ave_time_cost = 0.0

    model.eval()

    with torch.no_grad():
        for it, (images, disparity ,labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            disparity  = Variable(disparity ).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)            
            torch.cuda.synchronize()
            start_time = time.time()
            logits = model(images,disparity )   # for abalation study and our MAFNet
            torch.cuda.synchronize()

            
            end_time = time.time()
            if it>=5: # # ignore the first 5 frames
                ave_time_cost += (end_time-start_time)
            # convert tensor to numpy 1d array
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            # generate confusion matrix frame-by-frame
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1]) # conf is an n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            # save demo images
            visualize(image_name=names, predictions=logits.argmax(1), weight_name=args.weight_name)
            print("%s, %s, frame %d/%d, %s, time cost: %.2f ms, demo result saved."
                  %(args.model_name, args.weight_name, it+1, len(test_loader), names, (end_time-start_time)*1000))
 

    globalacc, pre_p, recall_p, F_score_p, iou_p, pre_b, recall_b, F_score_b, iou_b = getScores(conf_total)
    print ('Epoch {0:} glob acc : {1:.4f}, pre_p : {2:.4f}, recall_p : {3:.4f}, F_score_p : {4:.4f}, IoU_p : {5:.4f}'.format('pothole', globalacc*100, pre_p*100, recall_p*100, F_score_p*100, iou_p*100))
    print ('Epoch {0:} glob acc : {1:.4f}, pre_b : {2:.4f}, recall_b : {3:.4f}, F_score_b : {4:.4f}, IoU_b : {5:.4f}'.format('background', globalacc*100, pre_b*100, recall_b*100, F_score_b*100, iou_b*100))
    conf_total_matfile = os.path.join("./runs", 'conf_'+args.weight_name+'.mat')
    savemat(conf_total_matfile,  {'conf': conf_total}) # 'conf' is the variable name when loaded in Matlab
    print ('Average pre :{0:.4f}, recall: {1:.4f}, F_score: {2:.4f}, IoU : {3:.4f}'.format( (pre_b+pre_p)/2*100, (recall_b+recall_p)/2*100, (F_score_b+F_score_p)/2*100, (iou_b+iou_p)/2*100))
    print('\n* the average time cost per frame (with batch size %d): %.2f ms, namely, the inference speed is %.2f fps' %(batch_size, ave_time_cost*1000/(len(test_loader)-5), 1.0/(ave_time_cost/(len(test_loader)-5)))) # ignore the first 10 frames
    print('\n###########################################################################')

   
