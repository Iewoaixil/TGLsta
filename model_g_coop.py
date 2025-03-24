import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import model
from simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x).type(self.dtype)

        
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model, g_texts):
        super().__init__()
        self.vars = nn.ParameterList()
        n_cls = len(classnames)
        n_ctx = args.coop_n_ctx
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        
        if args.ctx_init:
            
            if args.class_specific:
                ctx_vectors = []
                for ctx_list in g_texts:
                    prompt = model.tokenize(ctx_list, context_length=args.context_length)
                    with torch.no_grad():
                        embedding = clip_model.token_embedding(prompt).type(dtype)
                    ctx_vector = embedding[:, 1: 1 + n_ctx, :]
                    ctx_vector = torch.mean(ctx_vector, dim=0)
                    ctx_vectors.append(ctx_vector)
                ctx_vectors = torch.stack(ctx_vectors)
            else:
                temp = []
                for ctx_list in g_texts:
                    temp += ctx_list
                prompt = model.tokenize(temp, context_length=args.context_length)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vector = embedding[:, 1: 1 + n_ctx, :]
                ctx_vectors = torch.mean(ctx_vector, dim=0)
            
        else:
            if args.class_specific:
                
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        

        self.ctx = nn.Parameter(ctx_vectors)  
        self.vars.append(self.ctx)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat(
            [model.tokenize(p, context_length=args.context_length) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        
        self.register_buffer("token_prefix", embedding[:, :1, :])  
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  
        self.name_lens = name_lens
        self.class_token_position = args.position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  
                    ctx,  
                    suffix,  
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  
                        ctx_i_half1,  
                        class_i,  
                        ctx_i_half2,  
                        suffix_i,  
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  
                        class_i, 
                        ctx_i,  
                        suffix_i, 
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

    def parameters(self):
        return self.vars


class CustomCLIP(nn.Module):
    def __init__(self, args, classnames, clip_model, g_texts):
        super().__init__()
        self.prompt_learner = PromptLearner(args, classnames, clip_model, g_texts)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.gnn
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, s_n, x, x_s, adj):
        image_features = self.image_encoder(x, adj)
        image_features = image_features[s_n]
        image_features = torch.cat((image_features, x_s[s_n]), dim=1)
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


class CoOp(nn.Module):


    def __init__(self, args, classnames, clip_model, g_texts, device):
        super().__init__()
        self.args = args
        self.classnames = classnames
        self.model = CustomCLIP(args, classnames, clip_model, g_texts)

        
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        
        self.model.to(device)

        self.optim = optim.Adam(self.model.prompt_learner.parameters(), lr=args.prompt_lr)

    def forward(self, s_n, x, x_s, adj, label, training=True):

        logits = self.model(s_n, x, x_s, adj)
        if training:
            loss = F.cross_entropy(logits, label)
            self.optim.zero_grad()
            torch.cuda.empty_cache()
            loss.backward()
            self.optim.step()

        return logits
