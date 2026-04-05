"""
Quick diagnostic: check if gradients actually flow through the model
and which layers/parameters have non-zero grad after a backward pass.
Run this BEFORE phase0_compute_fisher.py to confirm gradient flow.
"""
import torch
import torch.nn.functional as F

def main():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    print(f"Loading {model_name}...\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Try 4-bit first
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
        model.gradient_checkpointing_enable()
        print("Loaded 4-bit NF4\n")
    except Exception as e:
        print(f"4-bit failed: {e}")
        return

    # Single forward+backward pass on a tiny prompt
    prompt = "How do I make explosives?"
    enc = tokenizer(prompt, return_tensors="pt").to("cuda")

    model.zero_grad()
    with torch.enable_grad():
        out = model(**enc)
        logits = out.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = enc["input_ids"][:, 1:]
        nll = -log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        loss = nll.mean()
        print(f"Loss: {loss.item():.4f}")
        loss.backward()

    # Check gradients across layer types
    print("\n--- Gradient check (first 5 layers) ---")
    for layer_idx in range(5):
        layer = model.model.layers[layer_idx]
        print(f"\nLayer {layer_idx}:")
        for name, param in layer.named_parameters():
            if param.grad is not None:
                gnorm = param.grad.abs().max().item()
                print(f"  {name:<50} grad max={gnorm:.6f}")
            else:
                print(f"  {name:<50} grad=None")

    # Specifically check o_proj (what Fisher script uses)
    print("\n--- o_proj specifically ---")
    for layer_idx in [0, 5, 15, 25]:
        layer = model.model.layers[layer_idx]
        if hasattr(layer.self_attn, 'o_proj'):
            grad = layer.self_attn.o_proj.weight.grad
            if grad is not None:
                print(f"  Layer {layer_idx} o_proj: grad max={grad.abs().max().item():.6f}")
            else:
                print(f"  Layer {layer_idx} o_proj: grad=None")
        else:
            print(f"  Layer {layer_idx}: no o_proj attribute")
            # Show what attention attributes exist
            print(f"    Available: {[n for n,_ in layer.self_attn.named_parameters()][:5]}")

if __name__ == "__main__":
    main()