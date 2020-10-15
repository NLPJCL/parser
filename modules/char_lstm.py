# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class CharLSTM(nn.Module):
    r"""
    CharLSTM aims to generate character-level embeddings for tokens.
    It summerizes the information of characters in each token to an embedding using a LSTM layer.

    Args:
        n_char (int):
            The number of characters.
        n_embed (int):
            The size of each embedding vector as input to LSTM.
        n_out (int):
            The size of each output vector.
        pad_index (int):
            The index of the padding token in the vocabulary. Default: 0.
    """

    def __init__(self, n_chars, n_embed, n_out, pad_index=0):
        super().__init__()

        self.n_chars = n_chars
        self.n_embed = n_embed
        self.n_out = n_out
        self.pad_index = pad_index

        # the embedding layer
        self.embed = nn.Embedding(num_embeddings=n_chars,
                                  embedding_dim=n_embed)
        # the lstm layer
        self.lstm = nn.LSTM(input_size=n_embed,
                            hidden_size=n_out//2,
                            batch_first=True,
                            bidirectional=True)

    def __repr__(self):
        s = f"{self.n_chars}, {self.n_embed}, "
        s += f"n_out={self.n_out}, "
        s += f"pad_index={self.pad_index}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, x):
        r"""
        Args:
            x (~torch.Tensor): ``[batch_size, seq_len, fix_len]``.
                Characters of all tokens.
                Each token holds no more than `fix_len` characters, and the excess is cut off directly.
        Returns:
            ~torch.Tensor:
                The embeddings of shape ``[batch_size, seq_len, n_out]`` derived from the characters.
        """
        # [batch_size, seq_len, fix_len]
        mask = x.ne(self.pad_index)#不!=True
        # [batch_size, seq_len]
        lens = mask.sum(-1) #求批次中每个句子的真实的词的长度。
        char_mask = lens.gt(0)#[batch_size, seq_len]  #相当于普通mask    

        # [n, fix_len, n_embed] n:这一批次所有的词数， fix_len:所有的固定值 n_embed:词嵌入。
        x = self.embed(x[char_mask])# x[char_mask]: n 真实的词的个数 里面仍然会有pad的每个词中的字的...
        x = pack_padded_sequence(x, lens[char_mask], True, False)#lens[char_mask]得到每个词的长度
        x, (h, _) = self.lstm(x)
        #h:[2,n,d_out//2]#

        # [ n , d_out ]#
        h = torch.cat(torch.unbind(h), -1)#
        # [batch_size, seq_len, n_out]
        embed = h.new_zeros(*lens.shape, self.n_out)
        #char_mask: [batch_size, seq_len,1]   h:[ n , n_out ]
        #h:3,20
        embed = embed.masked_scatter_(char_mask.unsqueeze(-1), h)

        return embed
if __name__ == '__main__':
    cl = CharLSTM(5, 10, 20, 0)
    x = torch.randint(0, 2, (2, 3, 4))
    print(x)
    x = cl(x)
    print(x.size())
