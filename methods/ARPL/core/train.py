import torch
import torch.nn.functional as F
from torch.autograd import Variable
from methods.ARPL.arpl_utils import AverageMeter
from methods.ARPL.loss.LabelSmoothing import smooth_one_hot

from tqdm import tqdm


class FeatureTrainer:
    def __init__(self, net, criterion, optimizer, trainloader, use_gpu=True):
        """
        net: backbone feature extractor (frozen during classifier training)
        classifier: trainable classifier head (e.g. a small MLP or linear layer)
        criterion: loss function
        optimizer: optimizer for classifier
        trainloader: dataloader yielding (images, labels, idx)
        """
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.use_gpu = use_gpu

        # Feature storage
        self.features = []
        self.labels = []

    @torch.no_grad()
    def extract_features(self):

        self.net.eval()
        self.features = []
        self.labels = []

        for data, labels, _ in tqdm(self.trainloader):
            if self.use_gpu:
                data = data.cuda()
            
            feats, _ = self.net(data,True)  # only extract features
            self.features.append(feats.detach().cpu())
            self.labels.append(labels)

        self.features = torch.cat(self.features, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def train_classifier_epoch(self, epoch=None, args= None):
        """Train only the classifier using stored features."""
        self.net.train()
        losses = AverageMeter()
        torch.cuda.empty_cache()

        if len(self.features) == 0:
            raise ValueError("No features stored! Run extract_features() first.")

        feats = self.features.cuda() if self.use_gpu else self.features
        labels = self.labels.cuda() if self.use_gpu else self.labels


        batch_size = self.trainloader.batch_size
        n_batches = (len(feats) + batch_size - 1) // batch_size

        for i in tqdm(range(n_batches), desc=f"Classifier Epoch {epoch}"):
            start, end = i * batch_size, (i + 1) * batch_size
            x_batch = feats[start:end]
            y_batch = labels[start:end]

            self.optimizer.zero_grad()

            if args ==None:
                logits = self.net.module.resnet.fc(x_batch)
            else: 
                if args.model == 'timm_resnet50':
                    logits = self.net.module.resnet.fc(x_batch)
                elif args.model == 'timm_resnet50_repl':
                    logits = self.net.module.resnet.fc(x_batch)
                else:
                    logits = self.net.module.model.head(x_batch)

            _, loss = self.criterion(None, logits, y_batch)

            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), x_batch.size(0))

        print(f"[Classifier Epoch {epoch}] Loss: {losses.avg:.6f}")
        return losses.avg




def trainmixup(net, criterion, optimizer, trainloader, epoch=None, mixup=True, **options):
    net.train()
    losses = AverageMeter()

    torch.cuda.empty_cache()

    loss_all = 0
    for batch_idx, (data, labels, idx) in enumerate(tqdm(trainloader)):

        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()

            lam = torch.distributions.Beta(1, 1).sample().item()
            
            batch_size = data.size()[0]
            index = torch.randperm(batch_size).cuda() 
            
            data_mix = lam * data + (1 - lam) * data[index, :]
            labels_mix = labels[index]

            x, y = net(data, True)
            x_mix, y_mix = net(data_mix, True)


            xlist = [x,x_mix]
            ylist = [y,y_mix]

            labelslist = [labels,labels_mix,lam,index]

            logits, loss = criterion(xlist, ylist, labelslist)
            
            loss.backward()
            optimizer.step()


        losses.update(loss.item(), data.size(0))
        loss_all += losses.avg

    print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, len(trainloader), losses.val, losses.avg))

    return loss_all


def train(net, criterion, optimizer, trainloader, epoch=None, mixup=True, **options):
    net.train()
    losses = AverageMeter()

    torch.cuda.empty_cache()

    loss_all = 0
    for batch_idx, (data, labels, idx) in enumerate(tqdm(trainloader)):

        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            lam = None
            batch_size = data.size()[0]
            index = None
            x, y = net(data, True)
            xlist = [x,None]
            ylist = [y,None]

            labelslist = [labels,None,lam,index]

            logits, loss = criterion(xlist, ylist, labelslist)
            
            loss.backward()
            optimizer.step()
        
        losses.update(loss.item(), data.size(0))
        
        loss_all += losses.avg

    print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, len(trainloader), losses.val, losses.avg))

    return loss_all

def train_cs(net, netD, netG, criterion, criterionD, optimizer, optimizerD, optimizerG, 
        trainloader, epoch=None, **options):
    print('train with confusing samples')
    losses, lossesG, lossesD = AverageMeter(), AverageMeter(), AverageMeter()

    net.train()
    netD.train()
    netG.train()

    torch.cuda.empty_cache()
    
    loss_all, real_label, fake_label = 0, 1, 0
    for batch_idx, (data, labels, idx) in enumerate(tqdm(trainloader)):
        gan_target = torch.FloatTensor(labels.size()).fill_(0)
        if options['use_gpu']:
            data = data.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            gan_target = gan_target.cuda()
        
        data, labels = Variable(data), Variable(labels)
        
        noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).cuda()
        if options['use_gpu']:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)

        ###########################
        # (1) Update D network    #
        ###########################
        # train with real
        gan_target.fill_(real_label)
        targetv = Variable(gan_target)
        optimizerD.zero_grad()
        output = netD(data)
        errD_real = criterionD(output, targetv)
        errD_real.backward()

        # train with fake
        targetv = Variable(gan_target.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterionD(output, targetv)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        ###########################
        # (2) Update G network    #
        ###########################
        optimizerG.zero_grad()
        # Original GAN loss
        targetv = Variable(gan_target.fill_(real_label))
        output = netD(fake)
        errG = criterionD(output, targetv)

        # minimize the true distribution
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        errG_F = criterion.fake_loss(x).mean()
        generator_loss = errG + options['beta'] * errG_F
        generator_loss.backward()
        optimizerG.step()

        lossesG.update(generator_loss.item(), labels.size(0))
        lossesD.update(errD.item(), labels.size(0))


        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss
        optimizer.zero_grad()
        x, y = net(data, True, 0 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        _, loss = criterion(x, y, labels)

        # KL divergence
        noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).cuda()
        if options['use_gpu']:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        F_loss_fake = criterion.fake_loss(x).mean()
        total_loss = loss + options['beta'] * F_loss_fake
        total_loss.backward()
        optimizer.step()
    
        losses.update(total_loss.item(), labels.size(0))

        # if (batch_idx+1) % options['print_freq'] == 0:
        #     print("Batch {}/{}\t Net {:.3f} ({:.3f}) G {:.3f} ({:.3f}) D {:.3f} ({:.3f})" \
        #     .format(batch_idx+1, len(trainloader), losses.val, losses.avg, lossesG.val, lossesG.avg, lossesD.val, lossesD.avg))
    
        loss_all += losses.avg

    print("Batch {}/{}\t Net {:.3f} ({:.3f}) G {:.3f} ({:.3f}) D {:.3f} ({:.3f})" \
    .format(batch_idx+1, len(trainloader), losses.val, losses.avg, lossesG.val, lossesG.avg, lossesD.val, lossesD.avg))

    return loss_all
