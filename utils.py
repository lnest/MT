import re
import spacy
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
#from torchtext.datasets import TranslationDataset


def load_dataset(batch_size):
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')
    url = re.compile('(<url>.*</url>)')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]
    
    '''
    DE = Field(tokenize=tokenize_de, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    EN = Field(tokenize=tokenize_en, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    '''
    DE = Field(include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    EN = Field(include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    
    #import pdb;pdb.set_trace()

    train, val, test = Multi30k.splits(root=".data/", exts=('.de', '.en'), fields=(DE, EN))
    print (len(train), len(val), len(test))
    DE.build_vocab(train.src, min_freq=2)
    EN.build_vocab(train.trg, max_size=10000)
    train_iter, val_iter, test_iter = BucketIterator.splits(
            (train, val, test), batch_size=batch_size, repeat=False)
    return train_iter, val_iter, test_iter, DE, EN
