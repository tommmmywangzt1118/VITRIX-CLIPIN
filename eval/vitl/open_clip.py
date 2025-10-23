import os
import sys
import torch.nn.functional as F
parent_dir = os.path.dirname(os.path.abspath(__file__))
# 将父目录添加到 Python 的模块搜索路径
sys.path.append(parent_dir)

from open_clip_pkg import create_model_and_transforms, get_tokenizer,get_model_config,CustomTextCLIP
from copy import deepcopy
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将models目录添加到Python路径
sys.path.insert(0, current_dir)
from .transformer_rope import TextTransformerRoPE,precompute_freqs_cis_dynamic_ntk_scaling, _expand_token, text_global_pool

def load_open_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu",long_cap: bool=False,stage_1:bool=False,text_path:str=None,context_length: int=77):
    if long_cap:
        model,_,transform=create_model_and_transforms(model_name,force_custom_text=True)
        text_cfg=get_model_config(model_name)
        text=TextTransformerRoPE(context_length=context_length, 
                                vocab_size=text_cfg["text_cfg"]["vocab_size"],
                                width=text_cfg["text_cfg"]["width"],
                                heads=text_cfg["text_cfg"]["heads"],
                                layers=text_cfg["text_cfg"]["layers"],
                                output_dim=text_cfg["text_cfg"]["width"])
        tokenizer=get_tokenizer(model_name,context_length=context_length)
        text.to(device)
        model.text=deepcopy(text)
        ckpt=torch.load(pretrained)
        #FIXME change parameter name to fit the model
        new_state_dict = {}
        for k, v in ckpt['state_dict'].items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        missing_keys, unexpected_keys=model.load_state_dict(new_state_dict,strict=False)
        model.to(device)
        model.encode_text = encode_text.__get__(model, CustomTextCLIP)
        print(f"Missing keys {missing_keys}")
        print(f"unexpected keys {unexpected_keys}")
    elif stage_1:
        model,_,transform=create_model_and_transforms(model_name,pretrained=pretrained,force_custom_text=True)
        text_cfg=get_model_config(model_name)
        text=TextTransformerRoPE(context_length=context_length, 
                                vocab_size=text_cfg["text_cfg"]["vocab_size"],
                                width=text_cfg["text_cfg"]["width"],
                                heads=text_cfg["text_cfg"]["heads"],
                                layers=text_cfg["text_cfg"]["layers"],
                                output_dim=text_cfg["text_cfg"]["width"])
        tokenizer=get_tokenizer(model_name,context_length=context_length)
        ckpt=torch.load(text_path)
        ckpt['state_dict'] = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
        filtered_state_dict = {}
        for name, param in text.named_parameters():
                if name in ckpt['state_dict']:
                    filtered_state_dict[name] = ckpt['state_dict'][name]
                else:
                    print(f"Warning: {name} not found in checkpoint")

            # Load the filtered state dict
        missing_keys, unexpected_keys = text.load_state_dict(filtered_state_dict, strict=False)
        text.to(device)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        model.text=deepcopy(text)
        model.encode_text = encode_text.__get__(model, CustomTextCLIP)
        model.to(device)
    else:
        model, _, transform = create_model_and_transforms(model_name, pretrained=pretrained, cache_dir=cache_dir, force_custom_text=True,force_context_length=context_length)
        model = model.to(device)
        tokenizer = get_tokenizer(model_name)
    return model, transform, tokenizer

def encode_text(self, text, normalize: bool = False):
    cast_dtype = self.text.transformer.get_cast_dtype()
    seq_len = text.shape[1]
    x = self.text.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
    attn_mask = self.text.attn_mask
    if self.text.cls_emb is not None:
        seq_len += 1
        x = torch.cat([x, _expand_token(self.text.cls_emb, x.shape[0])], dim=1)
        cls_mask = self.build_cls_mask(text, cast_dtype)
        if attn_mask is not None:
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.text.transformer(x, attn_mask=attn_mask)
    x = x.permute(1, 0, 2)  # LND -> NLD

    # x.shape = [batch_size, n_ctx, transformer.width]
    if self.text.cls_emb is not None:
        # presence of appended cls embed (CoCa) overrides pool_type, always take last token
        pooled, tokens = text_global_pool(x, pool_type='last')
        pooled = self.text.ln_final(pooled)  # final LN applied after pooling in this case
    else:
        x = self.text.ln_final(x)
        pooled, tokens = text_global_pool(x, text, pool_type=self.text.pool_type)

    if self.text.text_projection is not None:
        if isinstance(self.text.text_projection, torch.nn.Linear):
            pooled = self.text.text_projection(pooled)
        else:
            pooled = pooled @ self.text.text_projection


    return F.normalize(pooled, dim=-1) if normalize else pooled