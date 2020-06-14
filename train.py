import argparse
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report

from models import BiLSTM
from utils import load_data, pad

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=30,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size per epoch.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--num_layers', type=int, default=2,
                    help='Number of recurrent layers.')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--embedding', type=int, default=128,
                    help='Number of word embedding dimension.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

train_word_lists, train_tag_lists, train_lengths, word2id, tag2id, train_word_size = load_data(
    dataset='train')

PAD = len(word2id)
UNK = PAD + 1
max_len = train_lengths[0]

train_word_id_lists, _ = pad(
    train_word_lists, train_lengths, max_len, word2id, PAD, UNK)
train_tag_id_lists, train_tag_ids = pad(train_tag_lists, train_lengths,
                                        max_len, tag2id, PAD, UNK)

dev_word_lists, dev_tag_lists, dev_lengths, _, _, dev_word_size = load_data(
    dataset='dev')

dev_word_id_lists, _ = pad(dev_word_lists, dev_lengths,
                           max_len, word2id, PAD, UNK)
dev_tag_id_lists, dev_tag_ids = pad(
    dev_tag_lists, dev_lengths, max_len, tag2id, PAD, UNK)

test_word_lists, test_tag_lists, test_lengths, _, _, test_word_size = load_data(
    dataset='test')

test_word_id_lists, _ = pad(test_word_lists, test_lengths,
                            max_len, word2id, PAD, UNK)
test_tag_id_lists, test_tag_ids = pad(test_tag_lists, test_lengths,
                                      max_len, tag2id, PAD, UNK)

train_vocab_size = len(train_word_id_lists)
dev_vocab_size = len(dev_word_id_lists)
embedding_dim = args.embedding
hidden_dim = args.hidden
num_layers = args.num_layers
dropout = args.dropout

# tag
out_dim = len(tag2id) + 2

model = BiLSTM(train_vocab_size, embedding_dim,
               num_layers, hidden_dim, dropout, out_dim)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

batch_size = args.batch_size

if args.cuda:
    model.cuda()
    train_word_id_lists = train_word_id_lists.cuda()
    train_tag_id_lists = train_tag_id_lists.cuda()
    dev_word_id_lists = dev_word_id_lists.cuda()
    dev_tag_id_lists = dev_tag_id_lists.cuda()
    test_word_id_lists = test_word_id_lists.cuda()
    test_tag_id_lists = test_tag_id_lists.cuda()

train_total_step = train_vocab_size // batch_size + 1
dev_total_step = dev_vocab_size // batch_size + 1

best_model = None
best_val_loss = float('inf')


def train(epoch):
    global best_val_loss
    global best_model

    model.train()
    optimizer.zero_grad()
    train_loss = 0.
    train_acc = 0
    train_pres = []

    for batch in range(0, train_vocab_size, batch_size):
        batch_x = train_word_id_lists[batch:batch + batch_size]
        batch_y = train_tag_id_lists[batch:batch + batch_size]
        lengths_batch = train_lengths[batch:batch + batch_size]

        # forward
        pres = model(batch_x, lengths_batch)

        # loss
        optimizer.zero_grad()
        loss, acc, pres = loss_acc(pres, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += acc
        train_pres += pres

    model.eval()
    with torch.no_grad():
        val_loss = 0.
        val_acc = 0
        val_pres = []
        for batch in range(0, dev_vocab_size, batch_size):
            batch_x = dev_word_id_lists[batch:batch + batch_size]
            batch_y = dev_tag_id_lists[batch:batch + batch_size]
            lengths_batch = dev_lengths[batch:batch + batch_size]

            # forward
            pres = model(batch_x, lengths_batch)

            # loss
            loss, acc, pres = loss_acc(pres, batch_y)

            val_loss += loss.item()
            val_acc += acc
            val_pres += pres

        if val_loss < best_val_loss:
            print("best model update...")
            best_model = deepcopy(model)
            best_val_loss = val_loss

    print("Train, Epoch {}, Loss: {:.4f}, Acc: {:.4f}".format(
        epoch, train_loss / train_total_step, train_acc / train_word_size))
    # print("Train, Classification report: \n", (classification_report(train_tag_ids, train_pres, labels=list(range(18)), target_names=list(tag2id.keys()))))
    # print("Train, F1 micro averaging:", (f1_score(train_tag_ids, train_pres, average='micro')))

    print("Val, Loss: {:.4f}, Acc: {:.4f}".format(
        val_loss / dev_total_step, val_acc / dev_word_size))
    # print("Val, Classification report: \n", (classification_report(dev_tag_ids, val_pres, labels=list(range(18)), target_names=list(tag2id.keys()))))
    # print("Val, F1 micro averaging:", (f1_score(dev_tag_ids, val_pres, average='micro')))


def loss_acc(pres, batch_y):
    mask = (batch_y != PAD)
    batch_y = batch_y[mask]
    pres = pres.masked_select(
        mask.unsqueeze(2).expand(-1, -1, out_dim)
    ).contiguous().view(-1, out_dim)

    loss = F.cross_entropy(pres, batch_y)
    pres = torch.max(pres, dim=1).indices
    acc = pres == batch_y
    acc = sum(acc).item()

    return loss, acc, pres.tolist()


for epoch in range(args.epochs):
    train(epoch)

with torch.no_grad():
    best_model.eval()
    pres = model(test_word_id_lists, test_lengths)

    # loss
    loss, acc, pres = loss_acc(pres, test_tag_id_lists)

    print("Test, Loss: {:.4f}, Acc: {:.4f}".format(
        loss.item(), acc / test_word_size))
    print("Test, Classification report: \n", (classification_report(
        test_tag_ids, pres, labels=list(range(18)), target_names=list(tag2id.keys()))))
    # print("Test, F1 micro averaging:", (f1_score(test_tag_ids, pres, average='micro')))

    torch.save(best_model, 'lstm.pt')
