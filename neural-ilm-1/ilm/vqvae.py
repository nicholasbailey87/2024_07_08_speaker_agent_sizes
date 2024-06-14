#!/usr/bin/env python
"""
vqvae, which we will use ilm on

start with plain vqvae...

so, we'll take in an image, pass through the network, then backprop...
"""
import os, json, time, datetime, argparse, math
from os import path

import torch
import torch.nn.functional as F
from torch import nn, optim, autograd
import numpy as np

from ulfs.stats import Stats
from ulfs.runner_base_v1 import RunnerBase
from ulfs import metrics, utils, profiling
from data_code.clevr.three_shapes_dataset import Dataset

def images_to_flat_for_codebook(images):
    """
    takes [N][C][H][W], converts to [N * H * W][E]
    """
    images_size = list(images.size())
    assert len(images_size) == 4
    images = images.transpose(1, 2).transpose(2, 3).contiguous().view(-1, images_size[1])
    return images_size, images

def images_for_codebook_to_4d(images_size, images_flat):
    images = images_flat.view(images_size[0], images_size[2], images_size[3], images_size[1]).transpose(2, 3).transpose(1, 2)
    return images

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        x_skip = x
        x = F.relu(x)
        x = F.relu(self.bn1(self.c1(x)))
        x = self.bn2(self.c2(x))
        x = x_skip + x
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()

        # 64
        self.c1 = nn.Conv2d(input_dim, embedding_dim, kernel_size=4, stride=2, padding=1)  # 32
        self.c2 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1)  # 16
        self.c3 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1)  # 8
        self.c4 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1)  # 4

        self.r1 = ResBlock(embedding_dim)
        self.r2 = ResBlock(embedding_dim)

        self.bn1 = nn.BatchNorm2d(embedding_dim)
        self.bn2 = nn.BatchNorm2d(embedding_dim)
        self.bn3 = nn.BatchNorm2d(embedding_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.c1(x)))
        x = F.relu(self.bn2(self.c2(x)))
        x = F.relu(self.bn3(self.c3(x)))
        x = self.c4(x)
        x = self.r1(x)
        x = self.r2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()

        self.r1 = ResBlock(embedding_dim)
        self.r2 = ResBlock(embedding_dim)

        self.d1 = nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1)
        self.d2 = nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1)
        self.d3 = nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1)
        self.d4 = nn.ConvTranspose2d(embedding_dim, input_dim, kernel_size=4, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(embedding_dim)
        self.bn2 = nn.BatchNorm2d(embedding_dim)
        self.bn3 = nn.BatchNorm2d(embedding_dim)

    def forward(self, x):
        x = self.r1(x)
        x = self.r2(x)
        x = F.relu(self.bn1(self.d1(x)))
        x = F.relu(self.bn2(self.d2(x)))
        x = F.relu(self.bn3(self.d3(x)))
        x = torch.tanh(self.d4(x))
        return x

class VQVAE(nn.Module):
    def __init__(self, num_latents, embedding_dim):
        super().__init__()
        self.codebook = nn.Parameter((torch.rand(num_latents, embedding_dim) * 0.2 - 0.1))

    def forward(self, x):
        """
        x assumed to be a matrix, with each example as a row
        """
        distances = metrics.calc_squared_euc_dist(x, self.codebook)
        _, Z = distances.min(dim=-1)
        q = self.codebook[Z]
        q_with_st = x + (q - x).detach()

        vq_loss = (q.detach() - x).pow(2).mean()
        commitment_loss = (x.detach() - q).pow(2).mean()

        return Z, q_with_st, (vq_loss, commitment_loss)

class EncoderPathway(nn.Module):
    def __init__(self, encoder, codebook):
        super().__init__()
        self.encoder = encoder
        self.codebook = codebook

    def forward(self, images):
        x = self.encoder(images)

    def sup_train_batch(self, images, latents):
        """
        assume that codebook already trained, so we only update the encoder
        """
        pass

class DecoderPathway(nn.Module):
    def __init__(self, Opt, lr, codebook, decoder):
        super().__init__()
        self.codebook = codebook
        self.decoder = decoder
        self.opt = Opt(lr=lr, params=self.parameters())
        self.crit = nn.MSELoss()

    def forward(self, latents):
        """
        latents assumed to be [N][H][W]
        """
        latents_d = list(latent.size())
        N = latents_d[0]
        latents_flat = latents.view(N, -1)
        print('latents_flat.size()', latents_flat.size())
        e_q_flat = self.codebook(latents_flat)
        print('e_q_flat.size()', e_q_flat.size())
        E = e_q_flat.size(-1)
        e_q = e_q_flat.view(*latents_d, E)
        print('e_q.size()', e_q.size())
        i_r = self.decoder(e_q)
        print('i_r.size()', i_r.size())
        return i_r

    def sup_train_batch(self, images, latents):
        """
        we will update both the decoder and also the codebook
        """
        N = images.size(0)
        assert N == latents.size(0)
        images_reconst = self(latents)
        loss = self.crit(images_reconst, images)

class Model(nn.Module):
    def __init__(self, p):
        super().__init__()
        Opt = getattr(optim, p.opt)
        self.encoder = Encoder(input_dim=3, embedding_dim=p.embedding_size)
        self.decoder = Decoder(input_dim=3, embedding_dim=p.embedding_size)
        self.vqvae = VQVAE(num_latents=p.num_latents, embedding_dim=p.embedding_size)
        self.encoder_pathway = EncoderPathway(encoder=self.encoder, codebook=self.vqvae.codebook)
        self.decoder_pathway = DecoderPathway(Opt=Opt, lr=p.lr, codebook=self.vqvae.codebook, decoder=self.decoder)

    def forward(self, x):
        encoded = self.encoder(x)
        d = list(encoded.size())
        encoded_flat = encoded.transpose(1, 2).transpose(2, 3).contiguous().view(-1, d[1])
        z_flat, encoded_q_flat, (vq_loss, commitment_loss) = self.vqvae(encoded_flat)
        z = z_flat.view(d[0], d[2], d[3])
        encoded_q = encoded_q_flat.view(d[0], d[2], d[3], -1).transpose(2, 3).transpose(1, 2).contiguous()

        reconst = self.decoder(encoded_q)
        return z, reconst, (vq_loss, commitment_loss)

class Runner(RunnerBase):
    def __init__(self):
        super().__init__(
            save_as_statedict_keys=['teacher'],
            additional_save_keys=[],
            step_key='training_step'
        )

    def setup(self, p):
        self.dataset = Dataset(data_dir=p.data_dir)
        self.teacher = Model(p=p)
        if self.enable_cuda:
            self.teacher = self.teacher.cuda()

        self.sup_train_N = int(self.dataset.N_train * p.sup_frac)
        print('sup_train_N', self.sup_train_N)

    def step(self, p):
        training_step = self.training_step
        step = self.training_step
        render = self.should_render()

        sup_images = self.dataset.sample_images(self.sup_train_N)
        if p.enable_cuda:
            sup_images = sup_images.cuda()

        with autograd.no_grad():
            sup_Z, _, _ = self.teacher(sup_images)
            sup_Z = sup_Z.detach()
            sup_Z_flat = sup_Z.view(-1)

        student = Model(p=p)
        if self.enable_cuda:
            student = student.cuda()
        _opt = optim.Adam(lr=0.001, params=student.parameters())

        if (p.sup_steps is not None and p.sup_steps > 0) and \
            (step > 0 or not p.train_e2e):

            _epoch = 0
            _last_print = time.time()
            _crit = nn.MSELoss()
            _sup_start = time.time()
            sup_stats = Stats([
                'episodes_count',
                'reconst_loss_sum',
                'loss_sum',
                'sender_acc_sum',
                'e2e_reconst_loss_sum',
                'e2e_reconst_loss_full_sum',
            ])
            # SUP
            # for (path_name)
            while True:
                epoch_loss_sum = 0
                reconst_loss_sum = 0
                sender_acc_sum = 0

                idxes = torch.from_numpy(np.random.choice(self.sup_train_N, p.batch_size, replace=False))
                b_images = sup_images[idxes]
                b_size = p.batch_size
                b_Z = sup_Z[idxes]
                b_Z_flat = b_Z.view(-1)
                if self.enable_cuda:
                    b_images, b_Z_flat = b_images.cuda(), b_Z_flat.cuda()
                b_encoded = student.encoder(b_images)
                b_encoded_size, b_encoded_flat = images_to_flat_for_codebook(b_encoded)
                distances = torch.clamp(metrics.calc_squared_euc_dist(b_encoded_flat, student.vqvae.codebook), min=1e-8).sqrt()
                probs = F.softmax(- distances, dim=-1)
                _, _pred = probs.max(dim=-1)
                prot_loss = - probs[np.arange(b_encoded_size[0] * b_encoded_size[2] * b_encoded_size[3]), b_Z_flat].log().mean()
                codebook_out_flat = student.vqvae.codebook[b_Z_flat]

                # log_std = math.log(p.gaussian_std)
                # print('log_std', log_std)
                mean = codebook_out_flat
                kl_loss = - (1 + 2 * math.log(p.gaussian_std) - mean * mean - p.gaussian_std * 2).mean() / 2
                codebook_out_flat = torch.randn(b_encoded_size[0] * b_encoded_size[2] * b_encoded_size[3], b_encoded_size[1], device=b_encoded.device
                    ) * p.gaussian_std + mean

                codebook_out = images_for_codebook_to_4d(b_encoded_size, codebook_out_flat)
                b_reconst = student.decoder(codebook_out)
                reconst_loss = _crit(b_reconst, b_images)

                x = b_encoded_flat
                q = codebook_out_flat
                vq_loss = (q.detach() - x).pow(2).mean()
                commitment_loss = (x.detach() - q).pow(2).mean()

                loss = prot_loss + reconst_loss + p.vq * vq_loss + p.commit * commitment_loss + kl_loss
                _opt.zero_grad()
                loss.backward()
                _opt.step()

                with autograd.no_grad():
                    _, e2e_images_reconst, (_, _) = student(b_images)
                    e2e_reconst_loss = _crit(e2e_images_reconst, b_images)

                    images, _ = self.dataset.sample_batch(batch_size=p.batch_size)
                    if self.enable_cuda:
                        images = images.cuda()
                    images = images[0]
                    _, e2e_images_reconst_full, (_, _) = student(images)
                    e2e_reconst_full_loss = _crit(e2e_images_reconst_full, images)

                sup_stats.episodes_count += 1
                sup_stats.loss_sum += loss.item()
                sup_stats.reconst_loss_sum += reconst_loss.item()
                sup_stats.sender_acc_sum += (_pred.view(-1) == b_Z_flat).float().mean().item()
                sup_stats.e2e_reconst_loss_sum += e2e_reconst_loss.item()
                sup_stats.e2e_reconst_loss_full_sum += e2e_reconst_full_loss.item()

                _epoch += 1
                _done_training = False
                if p.sup_steps is not None and _epoch >= p.sup_steps:
                    print('done sup training (reason: steps)')
                    _done_training = True
                if _done_training or time.time() - _last_print >= 30.0:
                    # SUP
                    # images_and_reconst = torch.stack([b_images[:10], b_reconst[:10]], dim=1).transpose(0, 1)
                    # utils.save_image_grid(f'html/image_dump_sup_{p.ref}.png', images_and_reconst, text=f'{p.ref}', text_size=12)

                    _elapsed_time = time.time() - _sup_start
                    _reconst_loss = sup_stats.reconst_loss_sum / sup_stats.episodes_count
                    _loss = sup_stats.loss_sum / sup_stats.episodes_count
                    _sender_acc = sup_stats.sender_acc_sum / sup_stats.episodes_count
                    _e2e_reconst_loss = sup_stats.e2e_reconst_loss_sum / sup_stats.episodes_count
                    _e2e_reconst_loss_full = sup_stats.e2e_reconst_loss_full_sum / sup_stats.episodes_count
                    print('kl_loss %.3e' % kl_loss)
                    log_dict = {
                        'record_type': 'sup',
                        'ilm_epoch': step,
                        'epoch': _epoch,
                        'sps': int(_epoch / _elapsed_time),
                        'elapsed_time': int(_elapsed_time),
                        'reconst_loss': _reconst_loss,
                        'e2e_reconst_loss': _e2e_reconst_loss,
                        'e2e_reconst_loss_full': _e2e_reconst_loss_full,
                        'loss': _loss,
                        'sender_acc': _sender_acc,
                    }
                    formatstr = (
                        'sup e={epoch} i={ilm_epoch} '
                        't={elapsed_time:.0f} '
                        'sps={sps:.0f} '
                        'reconst_loss={reconst_loss:.3e} '
                        'e2e_reconst_loss={e2e_reconst_loss:.3e} '
                        'e2e_reconst_loss_full={e2e_reconst_loss_full:.3e} '
                        'loss={loss:.3e} '
                        'sender_acc={sender_acc:.3f} '
                    )
                    self.print_and_log(log_dict, formatstr=formatstr)

                    sup_stats.reset()
                    _last_print = time.time()
                if _done_training:
                    print('done sup training')
                    break
            sup_time = time.time() - _sup_start
            sup_loss = _loss
            sup_reconst_loss = _reconst_loss
            sup_sender_acc = _sender_acc
        else:
            sup_time = 0
            sup_loss = 0
            sup_reconst_loss = 0
            sup_sender_acc = 0

        e2e_time = 0
        if p.train_e2e:
            last_print = time.time()
            _e2e_start = time.time()
            e2e_stats = Stats([
                'episodes_count',
                'e2e_reconst_loss_sum',
                'e2e_commitment_loss_sum',
                'e2e_vq_loss_sum',
                'e2e_holdout_rho_sum',
                'e2e_holdout_loss_sum',
            ])
            epoch = 0
            # _opt = optim.Adam(lr=0.001, params=student.parameters())
            _crit = nn.MSELoss()
            # E2E
            while True:
                images, labels = self.dataset.sample_batch(batch_size=p.batch_size)
                if self.enable_cuda:
                    images, labels = images.cuda(), labels.cuda()
                # we'll just throw away the extra images...
                images = images[0]
                Z, images_reconst, (vq_loss, commitment_loss) = student(images)
                reconst_loss = _crit(images_reconst, images)
                loss = reconst_loss + p.vq * vq_loss + p.commit * commitment_loss

                _opt.zero_grad()
                loss.backward()
                _opt.step()

                e2e_stats.e2e_reconst_loss_sum += reconst_loss.item()
                e2e_stats.e2e_commitment_loss_sum += commitment_loss.item()
                e2e_stats.e2e_vq_loss_sum += vq_loss.item()
                e2e_stats.episodes_count += 1

                _done_training = False
                if p.e2e_steps is not None and epoch >= p.e2e_steps:
                    print('reached target e2e step', epoch, ' => breaking')
                    _done_training = True
                if time.time() - last_print >= 30 or _done_training:
                    # E2E

                    holdout_ep_count = 0
                    holdout_loss_sum = 0
                    holdout_rho_sum = 0

                    self.teacher.eval()
                    for batch_image, batch_labels in self.dataset.iter_holdout(batch_size=p.batch_size):
                        batch_image = batch_image[0]
                        if p.enable_cuda:
                            batch_image = batch_image.cuda()
                            batch_labels = batch_labels.cuda()
                        with autograd.no_grad():
                            holdout_Z, holdout_images_reconst, (_, _) = student(batch_image)
                        holdout_loss_sum += _crit(holdout_images_reconst, batch_image).item()
                        b_N = Z.size(0)
                        Z = Z.view(b_N, -1)
                        holdout_rho_sum += metrics.topographic_similarity(Z, batch_labels)
                        holdout_ep_count += 1
                        break
                    holdout_rho = holdout_rho_sum / holdout_ep_count
                    holdout_loss = holdout_loss_sum / holdout_ep_count
                    self.teacher.train()

                    reconst_loss = e2e_stats.e2e_reconst_loss_sum / e2e_stats.episodes_count
                    commitment_loss = e2e_stats.e2e_commitment_loss_sum / e2e_stats.episodes_count
                    vq_loss = e2e_stats.e2e_vq_loss_sum / e2e_stats.episodes_count

                    images_and_reconst = torch.stack([batch_image[:10], holdout_images_reconst[:10]], dim=1).transpose(0, 1)
                    utils.save_image_grid(f'html/image_dump_e2e__{p.ref}_{step}_{epoch}.png', images_and_reconst, text=f'{p.ref}', text_size=12)

                    _elapsed_time = time.time() - _e2e_start
                    log_dict = {
                        'record_type': 'e2e',
                        'ilm_epoch': step,
                        'epoch': epoch,
                        'sps': int(epoch / _elapsed_time),
                        'elapsed_time': int(_elapsed_time),
                        'reconst_loss': reconst_loss,
                        'commitment_loss': commitment_loss,
                        'vq_loss': vq_loss,
                        # 'holdout_loss': holdout_loss,
                        # 'holdout_rho': holdout_rho,
                    }
                    formatstr = (
                        'e2e e={epoch} i={ilm_epoch} '
                        't={elapsed_time:.0f} '
                        'sps={sps:.0f} '
                        'reconst_loss={reconst_loss:.3e} '
                        'commitment_loss={commitment_loss:.3e} '
                        'vq_loss={vq_loss:.3e} '
                        # 'holdout_loss={holdout_loss:.3e} '
                        # 'holdout_rho={holdout_rho:.3e} '
                    )
                    self.print_and_log(log_dict, formatstr=formatstr)

                    e2e_stats.reset()
                    last_print = time.time()
                if _done_training:
                    break
                epoch += 1
            e2e_time = time.time() - _e2e_start
            e2e_loss = reconst_loss + vq_loss + commitment_loss
            e2e_reconst_loss = reconst_loss
            e2e_commitment_loss = commitment_loss
            e2e_vq_loss = vq_loss
            # e2e_holdout_loss = holdout_loss
            # e2e_holdout_rho = holdout_rho

            self.teacher = student

        if True:
            _dists = metrics.calc_squared_euc_dist(self.teacher.vqvae.codebook, self.teacher.vqvae.codebook)
            _dists = metrics.tri_to_vec(_dists)
            _dists = _dists.sort()[0]
            print('_dists[:20]', _dists[:20])
            print('_dists < 1e-4:', (_dists < 1e-4).int().sum().item())

            images_and_reconst = torch.stack([images[:10], images_reconst[:10]], dim=1).transpose(0, 1)
            utils.save_image_grid(f'html/image_dump_{p.ref}.png', images_and_reconst, text=f'{p.ref}', text_size=12)

            log_dict = {
                'type': 'ilm',
                'sps': int(step / (time.time() - self.start_time)),
                'elapsed_time': time.time() - self.start_time,
                'sup_time': sup_time,
                'sup_reconst_loss': sup_reconst_loss,
                'sup_loss': sup_loss,
                'sup_sender_acc': sup_sender_acc,
                'e2e_time': e2e_time,
                'e2e_reconst_loss': e2e_reconst_loss,
                'e2e_commitment_loss': e2e_commitment_loss,
                'e2e_vq_loss': e2e_vq_loss,
                # 'e2e_holdout_loss': e2e_holdout_loss,
                # 'e2e_holdout_rho': e2e_holdout_rho
            }

            formatstr = (
                'e={training_step} '
                't={elapsed_time:.0f} '
                'sps={sps:.0f} '
                # 'sup[rec_l={sup_reconst_loss:.3e},l={sup_loss:.3e},sender_acc={sup_sender_acc:.3f}] '
                # 'e2e[rec_l={e2e_reconst_loss:.3e},com_l={e2e_commitment_loss:.3e},vq_l={e2e_vq_loss:.3e},ho_l={e2e_holdout_loss:.3e},ho_rho={e2e_holdout_rho:.3e}] '
            )
            self.print_and_log(log_dict, formatstr=formatstr)

        if p.max_generations is not None and step + 1 >= p.max_generations:
            print('reached max generations', p.max_generations, '=> terminating')
            self.finish = True

if __name__ == '__main__':
    utils.clean_argv()
    runner = Runner()

    runner.add_param('--data-family', type=str, default='objects_gl')
    runner.add_param('--ds-ref', type=str, required=True)
    runner.add_param('--commit', type=float, default=0)
    runner.add_param('--vq', type=float, default=0)
    # runner.add_param('--kl', type=float, default=0)
    runner.add_param('--data-dir', type=str, default='~/data/{data_family}/{ds_ref}')
    runner.add_param('--sup-frac', type=float, default=0.4)
    runner.add_param('--sup-steps', type=int)
    runner.add_param('--gaussian-std', type=float, default=1)
    runner.add_param('--e2e-steps', type=int)
    runner.add_param('--max-generations', type=int)
    runner.add_param('--opt', type=str, default='RMSprop')
    runner.add_param('--lr', type=float, default=0.001)
    runner.add_param('--no-train-e2e', action='store_true')

    runner.add_param('--seed', type=int)
    runner.add_param('--batch-size', type=int, default=32)
    runner.add_param('--num-latents', type=int, default=100)
    runner.add_param('--embedding-size', type=int, default=50)
    runner.add_param('--dropout', type=float, default=0.5)

    runner.parse_args()
    utils.reverse_args(runner.params, 'no_train_e2e', 'train_e2e')
    runner.params.data_dir = runner.params.data_dir.format(**runner.params.__dict__)
    print('runner.params', runner.params)
    runner.setup_base()
    runner.run_base()
