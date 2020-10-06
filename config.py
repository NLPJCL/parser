

class Config(object):
    epochs = 100
    patience=100
    lr=2e-3

class EISNER_Config(Config):
    ftrain = 'data/conll09/train.conllx'
    fdev = 'data/conll09/dev.conllx'
    ftest = 'data/conll09/test.conllx'
    iterator=20
    shuffle=True
    exitor=10


class TAG_BIAFFINE_Config(Config):
    ftrain = 'data/ptb/train.conllx'
    fdev = 'data/ptb/dev.conllx'
    ftest = 'data/ptb/test.conllx'
    fembed = 'data/glove.6B.100d.txt'

    #
    use_gpu=True
    #
    batch_size=50

    #词嵌入
    n_embed =100
    n_tag_embed =50

 #   mix_dropout = .0,
 #   embed_dropout = .33,

    n_lstm_hidden = 400
    n_lstm_layers = 3
    lstm_dropout = 0.33
    n_mlp_arc = 500
    n_mlp_rel = 100

    mlp_dropout =0.33
    '''
    feat_pad_index = 0,
    pad_index = 0,
    unk_index = 1,
    '''


class NPCRF_BIAFFINE_Config(Config):

    ftrain='data/codt-banjiao/partial/train.conll'
    fdev='data/codt-banjiao/partial/dev.conll'
    ftest='data/codt-banjiao/partial/test.conll'
    fembed='data/embed.txt'
    partial=True

    '''
    非投影弧的全标注数据
    ftrain = 'data/ptb/train.conllx'
    fdev = 'data/ptb/dev.conllx'
    ftest = 'data/ptb/test.conllx'
    fembed = 'data/glove.6B.100d.txt'
    '''
    #
    use_gpu=True
    #
    batch_size=50

    #词嵌入
    n_embed =100
    n_tag_embed =50

 #   mix_dropout = .0,
 #   embed_dropout = .33,

    n_lstm_hidden = 400
    n_lstm_layers = 3
    lstm_dropout = 0.33
    n_mlp_arc = 500
    n_mlp_rel = 100

    mlp_dropout =0.33
    '''
    feat_pad_index = 0,
    pad_index = 0,
    unk_index = 1,
    '''


config = {
    'global_eisner': EISNER_Config,
    'tag_biaffine': TAG_BIAFFINE_Config,
    'npcrf_biaffine': NPCRF_BIAFFINE_Config
}