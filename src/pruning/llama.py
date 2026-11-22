import time

import torch
import torch.nn as nn

from sparsegpt import *
from modelutils import *
from quant import *

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False


# ====================================================================
# Layer-wise Sparsity Parsing Helpers (Ported from opt.py)
# ====================================================================

def _validate_value(v: float):
    if not (0.0 <= v < 1.0):
        raise ValueError(f"Sparsity must be in [0,1), got {v}.")

def _validate_sparsities(vals):
    for v in vals:
        _validate_value(v)

def _parse_layer_sparsities_inline(spec: str, num_layers: int):
    """
    Accepted syntax:
      1) CSV: '0.2,0.3,0.4,...'  (len == num_layers)
      2) tokens: '0:0.1,1:0.2,5-7:0.5,*:0.3'
         - single index 'i:v'
         - range 'a-b:v' (a and b inclusive)
         - default '*:v' (applied to all other layers)
    Returns: list of num_layers floats
    """
    parts = [t.strip() for t in spec.split(',') if t.strip()]

    # 1) try CSV
    try:
        vals = [float(x) for x in parts]
        if len(vals) == num_layers:
            _validate_sparsities(vals)
            return vals
    except Exception:
        pass  # not CSV: parse as tokens

    # 2) token syntax
    result = [None] * num_layers
    default_val = None
    for tok in parts:
        if ':' not in tok:
            raise ValueError(f"Invalid token '{tok}'. Expected CSV or 'i:v' / 'a-b:v' / '*:v'.")
        key, vstr = tok.split(':', 1)
        val = float(vstr.strip())
        _validate_value(val)
        key = key.strip()

        if key == '*':
            default_val = val
            continue

        if '-' in key:
            a, b = key.split('-', 1)
            a = int(a.strip()); b = int(b.strip())
            if a < 0 or b >= num_layers or a > b:
                raise ValueError(f"Invalid range '{key}' for num_layers={num_layers}.")
            for i in range(a, b + 1):
                result[i] = val
        else:
            idx = int(key)
            if idx < 0 or idx >= num_layers:
                raise ValueError(f"Invalid layer index '{idx}' for num_layers={num_layers}.")
            result[idx] = val

    if default_val is None and any(v is None for v in result):
        missing = [i for i, v in enumerate(result) if v is None]
        raise ValueError(f"Missing sparsity for layers {missing}; provide them or use '*:val'.")

    filled = [default_val if v is None else v for v in result]
    _validate_sparsities(filled)
    return filled

def _parse_layer_sparsities(spec: str, num_layers: int):
    return _parse_layer_sparsities_inline(spec, num_layers)

def _is_structured_nm(args) -> bool:
    return (args.prunen and args.prunen > 0) and (args.prunem and args.prunem > 0)

def _build_layer_sparsities_strict(args, num_layers: int):
    """
    Exactly one of --sparsity, --layer-sparsities or (--prunen & --prunem) must be given.
    Returns: (layer_sparsities_list, prune_mode)
    """
    has_glob = (args.sparsity is not None)
    has_list = (args.layer_sparsities is not None)
    has_nm   = _is_structured_nm(args)

    cnt = int(bool(has_glob)) + int(bool(has_list)) + int(bool(has_nm))
    if cnt != 1:
        raise ValueError("Specify exactly ONE of: --sparsity  OR  --layer-sparsities  OR  (--prunen & --prunem).")

    if has_nm:
        s = float(args.prunen) / float(args.prunem)
        _validate_value(s)
        return [s] * num_layers, "structured"

    if has_list:
        lst = _parse_layer_sparsities(args.layer_sparsities, num_layers)
        return lst, "unstructured"

    _validate_value(args.sparsity)
    return [float(args.sparsity)] * num_layers, "unstructured"


# ====================================================================
# Model Loading & Sequential Pruning Logic
# ====================================================================

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    
    # Llama 3/3.1 support (avoid KeyError)
    cache = {"i": 0, "attention_mask": None, "position_ids": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    
    # collect layer kwargs dynamically (.get() for safety)
    layer_kwargs = {"attention_mask": cache.get("attention_mask")}
    if cache.get("position_ids") is not None:
        layer_kwargs["position_ids"] = cache.get("position_ids")
    if cache.get("position_embeddings") is not None:
        layer_kwargs["position_embeddings"] = cache.get("position_embeddings")

    print("Ready.")

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gpts = {}
            for name in subset:
                if (
                    not (args.minlayer <= i < args.maxlayer and args.prune_only in name)
                ) == (not args.invert):
                    continue
                gpts[name] = SparseGPT(subset[name])
                if args.wbits < 16:
                    gpts[name].quantizer = Quantizer()
                    gpts[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=False, mse=False
                    )

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            for j in range(args.nsamples):
                # use the collected layer_kwargs
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
                
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print("Pruning ...")
                # per-layer sparsity
                sparsity = args.layer_sparsities_list[i]
                gpts[name].fasterprune(
                    sparsity,
                    prunen=args.prunen,
                    prunem=args.prunem,
                    percdamp=args.percdamp,
                    blocksize=args.blocksize,
                    group_bytes=args.group_bytes  # block size in bytes
                )
                gpts[name].free()

        for j in range(args.nsamples):
            # use the collected layer_kwargs
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev,  dataset: str, log_wandb: bool = False):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    
    # Llama 3/3.1 support (avoid KeyError)
    cache = {"i": 0, "attention_mask": None, "position_ids": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    
    # collect layer kwargs dynamically (.get() for safety)
    layer_kwargs = {"attention_mask": cache.get("attention_mask")}
    if cache.get("position_ids") is not None:
        layer_kwargs["position_ids"] = cache.get("position_ids")
    if cache.get("position_embeddings") is not None:
        layer_kwargs["position_embeddings"] = cache.get("position_embeddings")

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            si = args.layer_sparsities_list[i]
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][
                    int(W.numel() * si)
                ]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            # use the collected layer_kwargs
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
            
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f"{dataset}/perplexity": ppl.item()})

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="LlaMA model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--sparsity", type=float, default=None, help="Target sparsity")
    parser.add_argument(
        '--layer-sparsities', type=str, default=None,
        help=(
            "Per-layer sparsity spec. Formats:\n"
            "  - CSV list (len==#layers): '0.1,0.2,0.3,...'\n"
            "  - Key:Val tokens: '0:0.1,1:0.2,5-7:0.5,*:0.3'\n"
            "    (ranges inclusive, * is default)\n"
        )
    )
    parser.add_argument("--prunen", type=int, default=0, help="N for N:M pruning.")
    parser.add_argument("--prunem", type=int, default=0, help="M for N:M pruning.")
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        '--group_bytes', type=int, default=0,
        help='Unstructured pruning chunk size in bytes (0=per-weight). E.g., 256 or 8192.'
    )
    parser.add_argument(
        "--gmp", action="store_true", help="Whether to run the GMP baseline."
    )
    parser.add_argument(
        "--wbits", type=int, default=16, help="Whether to quantize as well."
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Prune all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Prune all layers with id < this."
    )
    parser.add_argument(
        "--prune_only",
        type=str,
        default="",
        help="Prune only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument("--save", type=str, default="", help="Path to saved model.")
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )

    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_llama(args.model)
    model.eval()

    num_layers = len(model.model.layers)

    has_glob = (args.sparsity is not None)
    has_list = (args.layer_sparsities is not None)
    has_nm   = (args.prunen and args.prunem)

    if has_glob or has_list or has_nm:
        args.layer_sparsities_list, args.prune_mode = _build_layer_sparsities_strict(args, num_layers)
        print(f"[INFO] num_layers={num_layers}")
        print(f"[INFO] prune_mode={args.prune_mode}")
        print(f"[INFO] first 5 sparsities: {args.layer_sparsities_list[:5]}")
        print(f"[INFO] last 5 sparsities : {args.layer_sparsities_list[-5:]}")
    else:
        args.layer_sparsities_list = None
        args.prune_mode = "none"
        print(f"[INFO] num_layers={num_layers}")
        print(f"[INFO] prune_mode=none (no pruning)")

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if (args.layer_sparsities_list is not None) and not args.gmp:
        tick = time.time()
        llama_sequential(model, dataloader, DEV)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'down_proj' in n:
                break
        print(time.time() - tick)

    for dataset in ["wikitext2", "ptb", "c4"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print("Dataset:", dataset)
        llama_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        model.save_pretrained(args.save)
