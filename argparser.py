import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Punctuation restoration')
    parser.add_argument('--name', default='punctuation-restore', type=str, help='name of run')
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
    parser.add_argument('--pretrained-model', default='roberta-large', type=str, help='pretrained language model')
    parser.add_argument('--sequence-length', default=256, type=int,
                        help='sequence length to use when preparing dataset (default 256)')
    parser.add_argument('--lr', default=5e-6, type=float, help='learning rate')
    parser.add_argument('--decay', default=0, type=float, help='weight decay (default: 0)')
    parser.add_argument('--batch-size', default=8, type=int, help='batch size (default: 8)')
    parser.add_argument('--epoch', default=10, type=int, help='total epochs (default: 10)')
    parser.add_argument('--save-path', default='out/', type=str, help='model and log save directory')

    args = parser.parse_args()
    return args