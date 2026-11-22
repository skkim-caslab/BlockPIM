import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *

DEBUG = False 
DEBUG2 = False
DEBUG3 = False

# Block granularity follows the deployed weight precision (INT8, 1 byte per
# weight), so --group_bytes 32 keeps or prunes 32 INT8 weights together --
# one 32-byte DRAM access / PIM MAC operand.
DEPLOYED_ELEM_BYTES = 1

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01, group_bytes: int =0, layer_name="unnamed_layer"
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        bytes_per_elem = DEPLOYED_ELEM_BYTES
        if group_bytes > 0 and prunen == 0:
            if group_bytes % bytes_per_elem != 0:
                raise ValueError("group_bytes must be divisible by the element size in bytes")
            group_size = group_bytes // bytes_per_elem
        else:
            group_size = 1

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    if group_size > 1:
                        is_conv1d = isinstance(self.layer, transformers.Conv1D)
                        # keep_mask: True = keep, False = prune
                        keep_mask = torch.zeros_like(tmp, dtype=torch.bool)
                        
                        # number of weights to keep
                        keep_elements_needed = int(tmp.numel() * (1 - sparsity))
                        current_keep = 0
                        
                        # rank all weights by saliency, most important first
                        flat_tmp = tmp.flatten()
                        sorted_tmp, sorted_indices = torch.sort(flat_tmp, descending=True)
                        
                        for val, idx in zip(sorted_tmp, sorted_indices):
                            if current_keep >= keep_elements_needed:
                                break
                            
                            r = int(idx // count)
                            c = int(idx % count)
                            
                            if is_conv1d:
                                # Conv1D: group along rows (elements of the same column)
                                group_start = (r // group_size) * group_size
                                g_size = min(group_size, self.rows - group_start)
                                mask_slice = keep_mask[group_start : group_start + g_size, c]
                            else:
                                # Linear: group along columns (elements of the same row)
                                group_start = (c // group_size) * group_size
                                g_size = min(group_size, count - group_start)
                                mask_slice = keep_mask[r, group_start : group_start + g_size]
                            
                            # number of newly kept weights
                            delta = (~mask_slice).sum().item()
                            if delta > 0:
                                mask_slice[:] = True
                                current_keep += delta
                        
                        # mask1 = ~keep_mask (True = prune)
                        mask1 = ~keep_mask
                        
                        # Debug: achieved sparsity
                        if DEBUG2:
                            actual_sparsity = mask1.sum().item() / mask1.numel()
                            print(f"Block {i1}-{i2}: Actual sparsity: {actual_sparsity:.4f} (target: {sparsity}, keep: {current_keep})")
                    else:
                        thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                        mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if DEBUG3:
            try:
                self.save_pruned_weights(layer_name, sparsity)
            except Exception as e:
                print(f"Warning: failed to save results: {e}")
                import traceback
                traceback.print_exc()

        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))


    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()

    def save_pruned_weights(self, layer_name, sparsity_level, all_results=None):
        """Append the pruning result of this layer (with a 30x30 weight preview) to one text file."""
        import os
        
        save_dir = './spresults'
        os.makedirs(save_dir, exist_ok=True)
        
        # all layers go to a single file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        single_filepath = f"./spresults/sparsity_results_{timestamp}.txt"
        
        W_final = self.layer.weight.data.clone()
        original_shape = self.layer.weight.shape
        
        # restore the original shape
        if isinstance(self.layer, transformers.Conv1D):
            if W_final.shape != original_shape:
                W_final = W_final.t()
        if isinstance(self.layer, nn.Conv2d):
            if W_final.shape != original_shape:
                W_final = W_final.reshape(original_shape)
        
        # move to CPU
        W_final_cpu = W_final.cpu()
        total_elements = W_final_cpu.numel()
        zero_elements = (W_final_cpu == 0).sum().item()
        actual_sparsity = zero_elements / total_elements
        
        try:
            with open(single_filepath, 'a') as f:  # append
                # layer separator
                f.write(f"\n{'='*80}\n")
                f.write(f"LAYER: {layer_name}\n")
                f.write(f"{'='*80}\n")
                
                f.write(f"Layer Type: {type(self.layer).__name__}\n")
                f.write(f"Shape: {original_shape}\n")
                f.write(f"Target Sparsity: {sparsity_level}\n")
                f.write(f"Actual Sparsity: {actual_sparsity:.6f}\n")
                f.write(f"Compression: {1-actual_sparsity:.1%}\n")
                f.write(f"Total/Zero/Non-zero: {total_elements:,} / {zero_elements:,} / {total_elements-zero_elements:,}\n")
                
                # non-zero statistics
                non_zero_weights = W_final_cpu[W_final_cpu != 0]
                if len(non_zero_weights) > 0:
                    f.write(f"\nNon-zero Stats:\n")
                    f.write(f"  Mean: {non_zero_weights.float().mean():.6f}\n")
                    f.write(f"  Std:  {non_zero_weights.float().std():.6f}\n")
                    f.write(f"  Min:  {non_zero_weights.float().min():.6f}\n")
                    f.write(f"  Max:  {non_zero_weights.float().max():.6f}\n")
                
                # 30x30 Weight Preview
                f.write(f"\nWEIGHT PREVIEW (30x30):\n")
                f.write(f"{'-'*60}\n")
                
                preview_size = min(30, W_final_cpu.shape[0], W_final_cpu.shape[1])
                preview = W_final_cpu[:preview_size, :preview_size]
                
                # print each row
                for i in range(preview_size):
                    row = preview[i]
                    # 6 decimals, zeros shown as '0.000000'
                    row_str = [f"{x:8.6f}" if abs(x) > 1e-6 else " 0.000000" for x in row]
                    non_zero_count = (row != 0).sum().item()
                    f.write(f"Row {i:2d} [{non_zero_count:2d}/30]: {' '.join(row_str)}\n")
                
                f.write(f"{'-'*60}\n")
                f.write(f"Sparsity in preview: {((preview == 0).float().mean()):.4f}\n")
                
                # extra results (when all_results is given)
                if all_results:
                    f.write(f"\nADDITIONAL RESULTS:\n")
                    for key, value in all_results.items():
                        f.write(f"  {key}: {value}\n")
            
            print(f"Saved to {single_filepath} (layer: {layer_name})")
            print(f"  Sparsity: {actual_sparsity:.4f}")
            
        except Exception as e:
            print(f"Save failed: {e}")
            return None
        
        return single_filepath