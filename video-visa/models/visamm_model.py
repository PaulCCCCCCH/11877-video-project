import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class ViSAMMModel(BaseModel):
    def name(self):
        return 'ViSAMMModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='AutoEncoderMM')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.reconst_only = opt.reconst_only
        self.latent_code_only = opt.latent_code_only
        self.decode_only = opt.decode_only
        self.pixel_feat_only = opt.pixel_feat_only
        self.use_audio = opt.use_audio
        self.loss_names = ['G_L1']

        if self.isTrain or self.reconst_only:
            self.visual_names = ['img', 'reconst']

        if self.latent_code_only:
            self.visual_names = ['latent_code']

        if self.decode_only:
            self.visual_names = ['reconst']

        if self.pixel_feat_only:
            self.visual_names = ['f01', 'f02', 'f03', 'f04']

        self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      opt.init_type, opt.init_gain, self.gpu_ids, opt.use_audio, opt.text_encoder)

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        self.img = input['img'].to(self.device)
        self.image_paths = input['im_path']
        self.transc = {k: v.to(self.device) for k, v in input['transc'].items()}

        self.audio = input['audio'].to(self.device) if 'audio' in input else None



    def forward(self):
        if self.isTrain or self.reconst_only:
            self.reconst = self.netG(self.img, self.transc, self.audio, 0)
            return self.reconst

        if self.latent_code_only:
            self.latent_code= self.netG(self.img, self.transc, self.audio, 1)
            return self.latent_code
        
        if self.decode_only:
            self.reconst = self.netG(self.img, self.transc, self.audio, 2)
            return self.reconst

        if self.pixel_feat_only:
            self.f01, self.f02, self.f03, self.f04 = self.netG(self.img, self.transc, self.audio, 3)


    def backward_G(self):
        self.loss_G_L1 = self.criterionL1(self.reconst, self.img) 
        self.loss_G = self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
