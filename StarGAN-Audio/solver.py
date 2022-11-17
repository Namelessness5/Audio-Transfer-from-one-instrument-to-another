from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from os.path import join, basename, dirname, split
import time
import datetime
from data_loader import to_categorical
import librosa
import utils
from tqdm import tqdm
import soundfile as sf

loss_func = nn.BCEWithLogitsLoss()
loss_func1 = torch.nn.MSELoss()


def add_sn(model):
    """谱归一化,传入模型实例即可"""
    for name, layer in model.named_children():
        model.add_module(name, add_sn(layer))
        if isinstance(model, (nn.Conv2d, nn.Linear)):
            return nn.utils.spectral_norm(model)
        else:
            return model
    return model


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, train_loader, test_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.sampling_rate = config.sampling_rate
        self.hop_length = config.hop_length

        # Model configurations.
        self.num_speakers = config.num_speakers
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.G_depth = config.G_depth
        self.D_depth = config.D_depth
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.weight_decay = config.weight_decay

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(num_speakers=self.num_speakers, repeat_num=self.G_depth)
        self.D = Discriminator(num_speakers=self.num_speakers, repeat_num=self.D_depth)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2], weight_decay=self.weight_decay)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2], weight_decay=self.weight_decay)
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
        self.g_lr = g_lr
        self.d_lr = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def sample_spk_c(self, size):
        spk_c = np.random.randint(0, self.num_speakers, size=size)
        spk_c_cat = to_categorical(spk_c, self.num_speakers)
        return torch.LongTensor(spk_c), torch.FloatTensor(spk_c_cat)

    def classification_loss(self, logit, target):
        """Compute softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

    def load_wav(self, wavfile):
        wav, _ = librosa.load(wavfile, sr=self.sampling_rate, mono=True)
        return wav  # TODO

    def train(self):
        """Train StarGAN."""
        # Set data loader.
        train_loader = self.train_loader
        data_iter = iter(train_loader)
        
        # spectral norm
        self.G = add_sn(self.G)
        self.D = add_sn(self.D)

        # Read a batch of testdata
        test_wavfiles = self.test_loader.get_batch_test_data(batch_size=4)
        test_wavs = [self.load_wav(wavfile) for wavfile in test_wavfiles]

        # Determine whether do copysynthesize when first do training-time conversion test.
        cpsyn_flag = [True, False][0]
        # f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            print("resuming step %d ..."% self.resume_iters)
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch labels.
            try:
                mc_real, spk_label_org, spk_c_org = next(data_iter)
            except:
                data_iter = iter(train_loader)
                mc_real, spk_label_org, spk_c_org = next(data_iter)

            mc_real.unsqueeze_(1) # (B, D, T) -> (B, 1, D, T) for conv2d

            # Generate target domain labels randomly.
            # spk_label_trg: int,   spk_c_trg:one-hot representation 
            spk_label_trg, spk_c_trg = self.sample_spk_c(mc_real.size(0)) 
            
            spk_c_org1 = spk_c_org.clone()
            spk_c_org1 = spk_c_org1.cpu()
            spk_c_trg1 = spk_c_trg.clone()
            spk_c_trg1 = spk_c_trg1.cpu()

            mc_real = mc_real.to(self.device)                         # Input mc.
            spk_label_org = spk_label_org.to(self.device)             # Original spk labels.
            spk_c_org = spk_c_org.to(self.device)                     # Original spk acc conditioning.
            spk_label_trg = spk_label_trg.to(self.device)             # Target spk labels for classification loss for G.
            spk_c_trg = spk_c_trg.to(self.device)                     # Target spk conditioning.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real mc feats.
            bs = mc_real.shape[0]
            D_input_false = torch.zeros(bs).to(self.device)
            D_input_true = 0.9 * torch.ones(bs).to(self.device)
            D_input_false = D_input_false.reshape((bs, 1)).to(self.device)
            D_input_true = D_input_true.reshape((bs, 1)).to(self.device)

            out_src, out_cls_spks = self.D(mc_real, spk_c_org)
            d_loss_real = loss_func(out_src, D_input_true)
            # d_loss_real = - torch.mean(out_src)
            d_loss_cls_spks = self.classification_loss(out_cls_spks, spk_label_org)
            
            # Compute loss with fake mc feats.
            mc_fake = self.G(mc_real, spk_c_trg)
            out_src, out_cls_spks = self.D(mc_fake.detach(), spk_c_trg)
            # d_loss_fake = torch.mean(out_src)
            d_loss_fake = loss_func(out_src, D_input_false)

            # Compute loss for gradient penalty.
            alpha = torch.rand(mc_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * mc_real.data + (1 - alpha) * mc_fake.data).requires_grad_(True)
                
            alpha = alpha.clone().cpu().detach().numpy()
            alpha = alpha.reshape(bs)
            tmp_label = torch.zeros((bs, self.num_speakers))
            for j in range(bs):
                tmp_label[j] += spk_c_org1[j] * alpha[j] + (1 - alpha[j]) * spk_c_trg1[j]
            tmp_label = tmp_label.to(self.device)
            tmp_label = tmp_label.requires_grad_(True)

            out_src, _ = self.D(x_hat, tmp_label)
                
            # out_src, _ = self.D(x_hat, spk_c_trg)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls_spks + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()
                
            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls_spks'] = d_loss_cls_spks.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_critic == 0:
                # Original-to-target domain.
                mc_fake = self.G(mc_real, spk_c_trg)
                out_src, out_cls_spks = self.D(mc_fake, spk_c_trg)
                g_loss_fake = loss_func(out_src, D_input_true)
                # g_loss_fake = - torch.mean(out_src)
                g_loss_cls_spks = self.classification_loss(out_cls_spks, spk_label_trg)

                # Target-to-original domain.
                mc_reconst = self.G(mc_fake, spk_c_org)
                g_loss_rec = loss_func1(mc_real, mc_reconst)

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls_spks
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls_spks'] = g_loss_cls_spks.item()


            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            if (i + 1) % self.sample_step == 0:
                sampling_rate = self.sampling_rate
                num_mcep=36
                frame_period=5
                with torch.no_grad():
                    for idx, wav in tqdm(enumerate(test_wavs)):

                        wav_name = test_wavfiles[idx]
                        # Method = "WORLD"
                        # wav_name = basename(test_wavfiles[idx])
                        # f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
                        # f0_converted = pitch_conversion(f0=f0,
                        #     mean_log_src=self.test_loader.logf0s_mean_src, std_log_src=self.test_loader.logf0s_std_src,
                        #     mean_log_target=self.test_loader.logf0s_mean_trg, std_log_target=self.test_loader.logf0s_std_trg)
                        # coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
                        # coded_sp_norm = (coded_sp - self.test_loader.mcep_mean_src) / self.test_loader.mcep_std_src
                        #
                        # coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1).to(self.device)
                        # conds = torch.FloatTensor(self.test_loader.spk_c_trg).to(self.device)
                        # # print(conds.size())
                        # coded_sp_converted_norm = self.G(coded_sp_norm_tensor, conds).data.cpu().numpy()
                        # coded_sp_converted = np.squeeze(coded_sp_converted_norm).T * self.test_loader.mcep_std_trg + self.test_loader.mcep_mean_trg
                        # coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                        # # decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
                        # wav_transformed = world_speech_synthesis(f0=f0_converted, coded_sp=coded_sp_converted,
                        #                                         ap=ap, fs=sampling_rate, frame_period=frame_period)

                        # Method == "CQT"
                        splited, angles = utils.Split_Convert_A(wav_name, rate=self.sampling_rate)
                        result = np.array([0])
                        for j in range(len(splited)):
                            tmp = torch.FloatTensor(splited[j].reshape((1, 1, splited[j].shape[0], splited[j].shape[1]))).to(self.device)
                            
                            conds = torch.FloatTensor(self.test_loader.spk_c_trg).to(self.device)
                            tmp = self.G(tmp, conds).data.cpu().numpy()
                            tmp = tmp.reshape((tmp.shape[2], tmp.shape[3]))
                            
                            # angle = 2 * np.pi * np.random.random_sample(tmp.shape) - np.pi
                            # print(angle.shape)
                            # for k in range(500):
                            #     S = tmp * np.exp(1j * angle)
                            #     tmp = librosa.icqt(S,  sr = self.sampling_rate, hop_length=self.hop_length)
                            #     angle = np.angle(librosa.cqt(tmp, sr = self.sampling_rate, hop_length=self.hop_length))
                            
                            angle = angles[j]
                            tmp = tmp * np.exp(1j * angle)
                            tmp = librosa.icqt(tmp, sr=self.sampling_rate, hop_length=self.hop_length)
                            
                            result = np.concatenate((result, tmp), axis=0)
                        sf.write(
                            join(self.sample_dir, str(i+1)+'-'+basename(wav_name).split('.')[0]+'-vcto-{}'.format(self.test_loader.trg_spk)+'.wav'), result, sampling_rate, "PCM_16")

                        # if cpsyn_flag:
                        #     wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=coded_sp,
                        #                                 ap=ap, fs=sampling_rate, frame_period=frame_period)
                        #     sf.write(join(self.sample_dir, 'cpsyn-'+wav_name), wav_cpsyn, sampling_rate, "PCM_16")
                    # cpsyn_flag = False

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.pth'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.pth'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0:
                g_lr = (self.g_lr / float(self.num_iters_decay))
                d_lr = (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

