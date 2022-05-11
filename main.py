from discriminator import Discriminator
from generator import Generator
from preprocessing import get_data
from train import train_epoch


if __name__ == '__main__':
    gen_model = Generator()
    disc_model = Discriminator()
    dataset = get_data()
    for i in range(10): train_epoch(gen_model, disc_model, dataset.shuffle(1024).batch(4), i)

