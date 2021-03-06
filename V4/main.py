import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import vdsr;import subpixel;import edsr
from dataset import DatasetFromHdf5

# Training settings
parser = argparse.ArgumentParser(description="PyTorch VDSR")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate. Default=0.1")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 4")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')

#vdsr,edsr
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")

#subpixel
parser.add_argument('--upscale_factor', default=4,type=int, required=False, help="super resolution upscale factor")


#
parser.add_argument('--train_set', default='train', type=str, help='path to train_set')
parser.add_argument('--model', default='vdsr', type=str, help='available network: vdsr;edsr;subpixel')
parser.add_argument('--inchannel', default=9, type=int, help='channel number of the input')
parser.add_argument('--outchannel', default=9, type=int, help='channel number of the input')

def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True
        
    print("===> Loading datasets")
    train_set = DatasetFromHdf5("data/"+opt.train_set+".h5")
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    print("===> Building model"+opt.model)
    if opt.model == 'vdsr':
        model = vdsr.Net(in_channels=opt.inchannel,out_channels=opt.outchannel)
        criterion = nn.MSELoss(size_average=False)
    elif opt.model == 'edsr':
        #lr = 1e-4
        model = edsr.Net(in_channels=opt.inchannel,out_channels=opt.outchannel)
        criterion = nn.L1Loss(size_average=False)
    else:
        #lr=0.01
        model = subpixel.Net(upscale_factor=opt.upscale_factor)
        criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    
    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))  
            
    print("===> Setting Optimizer")
    if opt.model == 'vdsr':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
            
    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):        
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)
        

def total_gradient(parameters):
    """Computes a gradient clipping coefficient based on gradient norm."""
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    totalnorm = 0
    for p in parameters: 
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = totalnorm ** (1./2)
    return totalnorm
    
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    
    model.train()    

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
        
        loss = criterion(model(input), target)
        optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_norm(model.parameters(),opt.clip) 
        optimizer.step()
        
#        print(iteration)
        if iteration%50 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))
    
def save_checkpoint(model, epoch):
    model_out_path = "model/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()