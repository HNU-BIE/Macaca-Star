import torch
import itertools
import random
from .base_model import BaseModel
from . import networks3D, ssim

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'ssim_A', 'cor_coe_GA','D_B', 'G_B', 'cycle_B', 'idt_B','ssim_B','cor_coe_GB','G_A_freq']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B','D_B_freq']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # load/define networks
        self.netG_A = networks3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,   # nc number channels
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, blocks=[3,2,3],ismerge=False)
        self.netG_B = networks3D.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, blocks=[4,0,4],ismerge=False)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            print('use_sigmoid:'+str(use_sigmoid))
            self.netD_A = networks3D.define_D(opt.output_nc, opt.ndf, opt.netD,opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks3D.define_D(opt.input_nc, opt.ndf, opt.netD,opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B_freq = networks3D.define_WTD(opt.input_nc, opt.ndf, opt.init_type,opt.init_gain, self.gpu_ids)
        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_B_freq_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks3D.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_freq = torch.optim.Adam(self.netD_B_freq.parameters(), lr=opt.lr * 0.5,betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_D_freq)
            ssim_loss = ssim.SSIM3D(window_size=5)
            self.criterionSSIM = lambda pred, target: 1 - ssim_loss(pred, target)


    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input[0 if AtoB else 1].to(self.device)
        self.real_B = input[1 if AtoB else 0].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)

        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):

        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_D_B_freq(self):
        fake_B = self.fake_B_freq_pool.query(self.fake_B)
        self.loss_D_B_freq = self.backward_D_basic(self.netD_B_freq, self.real_B, fake_B)


    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_ssim = self.opt.lambda_ssim
        lambda_freq = self.opt.lambda_freq

        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        self.loss_G_A_spatial = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_A_freq = self.criterionGAN(self.netD_B_freq(self.fake_B), True)
        self.loss_G_A = self.loss_G_A_spatial + lambda_freq * self.loss_G_A_freq

        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # SSIM loss
        if lambda_ssim > 0:
            real_A_01 = (self.real_A + 1) * 0.5
            rec_A_01 = (self.rec_A + 1) * 0.5
            real_B_01 = (self.real_B + 1) * 0.5
            rec_B_01 = (self.rec_B + 1) * 0.5
            self.loss_ssim_A = self.criterionSSIM(rec_A_01, real_A_01) * lambda_ssim
            self.loss_ssim_B = self.criterionSSIM(rec_B_01, real_B_01) * lambda_ssim
        else:
            self.loss_ssim_A = 0
            self.loss_ssim_B = 0

        # Combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_ssim_A + self.loss_ssim_B
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_B_freq], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

        # Update Frequency Discriminator (D_B_freq)
        self.set_requires_grad([self.netD_B_freq], True)
        self.optimizer_D_freq.zero_grad()
        self.backward_D_B_freq()
        self.optimizer_D_freq.step()

