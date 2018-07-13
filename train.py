import os
import math
import argparse
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from utils import load_dataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import codecs

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='initial learning rate')
    p.add_argument('-rs', '--resume', default='', 
                   help='previous reader file name (in `model_dir`). ')
    return p.parse_args()

def write_records(records):
    with codecs.open(u"model_res_sample.txt", u"ab", u"utf-8") as w:
        for tuple in records:
            for item in tuple:
                w.write(item+u"\n")
            w.write(u"-"*80+u"\n")

def evaluate(model, val_iter, vocab_size, DE, EN, debug=False):
    model.eval()
    pad = EN.vocab.stoi['<pad>']
    total_loss = 0
    records = []
    for b, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src_cpu = src.transpose(0, 1).data.cpu()
        trg_cpu = trg.transpose(0, 1).data.cpu()
        src_mask = torch.eq(src.transpose(1, 0), pad)
        
        src_mask[:, 0] = 1

        for ri in range(len(src_cpu)):
            src_rec_text = []
            for wid in src_cpu[ri]:
                if wid == DE.vocab.stoi['<eos>']:
                    break
                else:
                    src_rec_text.append(DE.vocab.itos[wid])
            src_rec_text = u" ".join(src_rec_text)
            trg_rec_text = []
            for wid in trg_cpu[ri]:
                if wid == EN.vocab.stoi['<eos>']:
                    break
                else:
                    trg_rec_text.append(EN.vocab.itos[wid])
            trg_rec_text = u" ".join(trg_rec_text)
            records.append([src_rec_text, trg_rec_text])
                      
        src = Variable(src.data.cuda(), volatile=True)
        trg = Variable(trg.data.cuda(), volatile=True)
        de_input, output, hidden, att, attn_energies, enc_out = model(src, trg, src_mask, teacher_forcing_ratio=0.0)
        loss = F.cross_entropy(output[1:].view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
        
        total_loss += loss.data[0]
        if debug:
            np.save("de_input.npy", de_input.data.cpu().numpy())
            np.save("attn_energies.npy", attn_energies.data.cpu().numpy())
            np.save("enc_out.npy", enc_out.data.cpu().numpy())
            np.save("hidden.npy", hidden.data.cpu().numpy())
            np.save("att.npy", att.data.cpu().numpy())
            for id, rec in enumerate(att):
                plt.matshow(rec.data.cpu().numpy())
                plt.savefig(u"img/"+str(b)+u"_"+str(id)+u".png")
                
            p, wids = torch.max(output.transpose(0, 1), dim=-1)
            wids_cpu = wids.data.cpu()
            for ri in range(len(wids)):
                pred = []
                for wid in wids_cpu[ri]:
                    if wid == EN.vocab.stoi['<eos>']:
                        break
                    else:
                        pred.append(EN.vocab.itos[wid])
                pred = u" ".join(pred)
                records[ri].append(pred)
            write_records(records)
            records = []
                
            np.save("pred_word.npy", wids.data.cpu().numpy())
            
    return total_loss / len(val_iter)


def train(e, model, optimizer, train_iter, vocab_size, grad_clip, DE, EN, debug=False):
    model.train()
    total_loss = 0
    pad = EN.vocab.stoi['<pad>']
    for b, batch in enumerate(train_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src_mask = torch.eq(src.transpose(1, 0), pad)#[B, L]
        
        src_mask[:, 0] = 1

        src, trg = src.cuda(), trg.cuda()
        optimizer.zero_grad()
        de_input, output, hidden, att, attn_energies, enc_out = model(src, trg, src_mask, teacher_forcing_ratio=0.0)
        
#         np.save("debug/att/attn_energies_"+str(b)+".npy", attn_energies.data.cpu().numpy())
#         np.save("debug/att/att_"+str(b)+".npy", att.data.cpu().numpy())
#         np.save("debug/enc/enc_out_"+str(b)+".npy", enc_out.data.cpu().numpy())
#         np.save("debug/hid/hidden_"+str(b)+".npy", hidden.data.cpu().numpy())
#         np.save("debug/mask/mask_"+str(b)+".npy", src_mask.data.cpu().numpy())
        if debug:
            print (u"iter :", b)
        '''
        print ("pred size :", output[1:].size(), u"\t trg size :", trg[1:].size())
        import numpy as np
        np.save("pred.npy", output[1:].cpu().detach().numpy())
        np.save("trg.npy", trg[1:].cpu().numpy())
        '''
        loss = F.cross_entropy(output[1:].view(-1, vocab_size),
                               trg[1:].contiguous().view(-1),
                               ignore_index=pad)
        #print (u"loss :", loss)
        loss.backward()
        clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data[0]

        if b % 100 == 0 and b != 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                  (b, total_loss, math.exp(total_loss)))
            total_loss = 0
#         if b == 10:
#             break


def main():
    debug = False
    args = parse_arguments()
    hidden_size = 256
    embed_size = 300
    assert torch.cuda.is_available()

    print("[!] preparing dataset...")
    train_iter, val_iter, test_iter, DE, EN = load_dataset(args.batch_size)
    de_size, en_size = len(DE.vocab), len(EN.vocab)
            
    print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
          % (len(train_iter), len(train_iter.dataset),
             len(test_iter), len(test_iter.dataset)))
    print("[DE_vocab]:%d [en_vocab]:%d" % (de_size, en_size))

    print("[!] Instantiating models...")
    encoder = Encoder(de_size, embed_size, hidden_size,
                      n_layers=1, dropout=0.3, debug=debug)
    decoder = Decoder(embed_size, hidden_size, en_size,
                      n_layers=1, dropout=0.3, debug=debug)
    seq2seq = Seq2Seq(encoder, decoder, debug=debug).cuda()
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)
    
    model_dir = u".save/"
    if args.resume:
        seq2seq.load_state_dict(torch.load(os.path.join(model_dir, args.resume)))
        print (u"--- model loaded")
        val_loss = evaluate(seq2seq, val_iter, en_size, DE, EN, debug=debug)
        print (u"cur model val loss :", val_loss)
    else:
        print (u"--- brand new model")

    best_val_loss = None
    for e in range(1, args.epochs+1):
        train(e, seq2seq, optimizer, train_iter,
              en_size, args.grad_clip, DE, EN, debug=debug)
        val_loss = evaluate(seq2seq, val_iter, en_size, DE, EN, debug=debug)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
              % (e, val_loss, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model...")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(seq2seq.state_dict(), './.save/seq2seq_%d.pt' % (e))
            best_val_loss = val_loss
    test_loss = evaluate(seq2seq, test_iter, en_size, DE, EN)
    print("[TEST] loss:%5.2f" % test_loss)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
