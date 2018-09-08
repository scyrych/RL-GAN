import torch
from input_wrapper import InputWrapper
import gym
from generator_model import Generator
from discriminator_model import Discriminator
import random
import numpy as np
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

BATCH_SIZE = 16
LEARNING_RATE = 0.0001
LATENT_VECTOR_SIZE = 100
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000

log = gym.logger
log.set_level(gym.logger.INFO)


def iterate_batches(gym_envs, batch_size=BATCH_SIZE):
    batch = [e.reset() for e in gym_envs]
    env_gen = iter(lambda: random.choice(gym_envs), None)

    while True:
        e = next(env_gen)
        obs, _, is_done, _ = e.step(e.action_space.sample())

        if np.mean(obs) > 0.01:
            batch.append(obs)

        if len(batch) == batch_size:
            yield torch.Tensor(np.array(batch, dtype=np.float32))
            batch.clear()

        if is_done:
            e.reset()


def generate_extra_fake_samples():
    global batch_vec, gen_output_vec
    gen_input_vec = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1).normal_(0, 1).to(device)
    batch_vec = batch_vec.to(device)
    gen_output_vec = net_gener(gen_input_vec)


def train_discriminator():
    dis_optimizer.zero_grad()
    dis_output_true_vec = net_discr(batch_vec)
    dis_output_false_vec = net_discr(gen_output_vec.detach())
    dis_loss = objective(dis_output_true_vec, true_labels_vec) + objective(dis_output_false_vec, false_labels_vec)
    dis_loss.backward()
    dis_optimizer.step()
    dis_losses.append(dis_loss.item())


def train_generator():
    gen_optimizer.zero_grad()
    dis_output_vec = net_discr(gen_output_vec)
    gen_loss = objective(dis_output_vec, true_labels_vec)
    gen_loss.backward()
    gen_optimizer.step()
    gen_losses.append(gen_loss.item())


def create_tensorboard_report():
    global gen_losses, dis_losses, iter_number
    iter_number += 1

    if iter_number % REPORT_EVERY_ITER == 0:
        log.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e", iter_number, np.mean(gen_losses), np.mean(dis_losses))
        writer.add_scalar("gen_loss", np.mean(gen_losses), iter_number)
        writer.add_scalar("dis_loss", np.mean(dis_losses), iter_number)
        gen_losses = []
        dis_losses = []

    if iter_number % SAVE_IMAGE_EVERY_ITER == 0:
        writer.add_image("fake", vutils.make_grid(gen_output_vec.data[:64]), iter_number)
        writer.add_image("real", vutils.make_grid(batch_vec.data[:64]), iter_number)


if __name__ == '__main__':
    device = torch.device("cuda")
    envs = [InputWrapper(gym.make(name)) for name in ('Breakout-v0', 'AirRaid-v0', 'Pong-v0')]
    input_shape = output_shape = envs[0].observation_space.shape

    net_discr = Discriminator(input_shape).to(device)
    net_gener = Generator(output_shape).to(device)

    writer = SummaryWriter()
    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(params=net_gener.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(params=net_discr.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    gen_losses = []
    dis_losses = []
    iter_number = 0

    true_labels_vec = torch.ones(BATCH_SIZE, dtype=torch.float32, device=device)
    false_labels_vec = torch.zeros(BATCH_SIZE, dtype=torch.float32, device=device)

    for batch_vec in iterate_batches(envs):
        generate_extra_fake_samples()
        train_discriminator()
        train_generator()
        create_tensorboard_report()
