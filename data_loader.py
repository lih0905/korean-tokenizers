
import torch
from torchtext import data


def dataset(tokenizer, max_vocab_size):
    
    TEXT = data.Field(tokenize=tokenizer, include_lengths = True)
    LABEL = data.LabelField(dtype = torch.float)
    fields = {'text': ('text',TEXT), 'label': ('label',LABEL)}
    
    train_data, test_data = data.TabularDataset.splits(
                                path = 'data',
                                train = 'train_data.csv',
                                test = 'test_data.csv',
                                format = 'csv',
                                fields = fields,  
    )

    TEXT.build_vocab(train_data,
                    max_size = max_vocab_size,
    )
    LABEL.build_vocab(train_data)

    return TEXT, LABEL, train_data, test_data 

    
def dataloader(tokenizer, max_vocab_size, batch_size, device):

    TEXT, LABEL, train_data, test_data = dataset(tokenizer, max_vocab_size)

    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, test_data),
        batch_size = batch_size,
        sort_key = lambda x: len(x.text),
        sort_within_batch = True,
        device = device,
    )

    return TEXT, LABEL, train_iterator, test_iterator