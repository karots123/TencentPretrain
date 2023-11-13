import torch.nn as nn
import torch
from tencentpretrain.layers.layer_norm import LayerNorm


class Embedding(nn.Module):
    def __init__(self, args):
        super(Embedding, self).__init__()
        self.embedding_name_list = []
        self.dropout = nn.Dropout(args.dropout)
        self.remove_embedding_layernorm = args.remove_embedding_layernorm
        if not self.remove_embedding_layernorm and "dual" not in args.embedding:
            self.layer_norm = LayerNorm(args.emb_size)

    def update(self, embedding, embedding_name):
        setattr(self, embedding_name, embedding)
        self.embedding_name_list.append(embedding_name)

    def forward(self, src, seg):
        if self.embedding_name_list[0] == "dual":
            return self.dual(src, seg)

        for i, embedding_name in enumerate(self.embedding_name_list):
            embedding = getattr(self, embedding_name)

            if i == 0:
                emb = embedding(src, seg)
            else:
                emb = embedding(src, seg) + emb.clone()

        if not self.remove_embedding_layernorm:
            emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb

class EmbeddingPipe(torch.nn.Module):
    def __init__(self, args,model):
        super(EmbeddingPipe, self).__init__()
        self.word_embeddings = model.embedding
        self.mask = args.mask
        self.rotary_position_embedding=args.rotary_position_embedding
        self.relative_position_embedding=args.relative_position_embedding
        if self.relative_position_embedding:
            from tencentpretrain.layers.relative_position_embedding import RelativePositionEmbedding
            self.relative_pos_emb = RelativePositionEmbedding(bidirectional=True, heads_num=args.heads_num,
                                                              num_buckets=args.relative_attention_buckets_num)
        elif self.rotary_position_embedding:
            from tencentpretrain.utils.rope import precompute_freqs_cis
            self.freqs_cis = precompute_freqs_cis(args.hidden_size // args.heads_num, args.max_seq_length * 2)
        self.has_residual_attention = args.has_residual_attention
   
    def forward(self, ipt):
       
        src, tgt, seg=ipt
        self.devicde=src.device
        
        emb=self.word_embeddings(src, seg)
       
        hidden = emb

        return hidden,tgt,seg