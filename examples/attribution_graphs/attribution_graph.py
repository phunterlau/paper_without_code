# --- Core Idea Framing ---
# This script prototypes the idea of "Attribution Graphs" by:
# 1. Using Grad*Activation as a PROXY for the influence/contribution of specific
#    component activations (embeddings, layer outputs) on a target output (predicted token logit).
# 2. Visualizing these scores as a DIRECTED GRAPH along the sequential forward pathway:
#    embeddings → L0_attn → L0_mlp → L1_attn → L1_mlp → … → target.
# 3. Edge weights represent the calculated score of the SOURCE node, visualizing
#    the influence magnitude flowing *out* along this path.
# This is a simplification; true attribution graphs might involve more complex
# decomposition methods (e.g., path patching) to isolate direct contributions.
# ---

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import networkx as nx # Requires: pip install networkx
import matplotlib.pyplot as plt # Requires: pip install matplotlib
import gc
import sys
import time

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" # Or "gpt2", etc.
# MODEL_NAME = "gpt2" # Smaller model for testing
PROMPT = "The biological analogy for attribution graphs suggests that information flow can be traced through"

# --- Robust Device and Dtype Selection ---
if torch.cuda.is_available():
    DEVICE = "cuda"
    # Prefer bfloat16 on Ampere+ GPUs, fallback to float16
    try:
        # Test bf16 support
        torch.tensor([1.0], dtype=torch.bfloat16).to(DEVICE)
        TORCH_DTYPE = torch.bfloat16
        DTYPE_NAME = "bfloat16"
    except Exception:
        TORCH_DTYPE = torch.float16
        DTYPE_NAME = "float16"
elif sys.platform == "darwin" and torch.backends.mps.is_available():
    DEVICE = "mps"
    TORCH_DTYPE = torch.float16 # bfloat16 support on MPS is limited/experimental
    DTYPE_NAME = "float16"
    if not torch.backends.mps.is_built():
        print("WARNING: MPS available but not built? Falling back to CPU.")
        DEVICE = "cpu"
        TORCH_DTYPE = torch.float32
        DTYPE_NAME = "float32"
else:
    DEVICE = "cpu"
    TORCH_DTYPE = torch.float32 # Use float32 on CPU
    DTYPE_NAME = "float32"

print(f"Selected Device: {DEVICE.upper()}")
print(f"Selected Dtype: {DTYPE_NAME}")
print(f"PyTorch version: {torch.__version__}")


# --- Globals for Hook Management ---
activations = {}
gradients = {}
hook_handles = []

# --- Hook Helper Functions (Streamlined) ---
def clear_hooks_data():
    """Clears stored activation and gradient data."""
    activations.clear()
    gradients.clear()
    gc.collect()

def remove_all_hooks():
    """Removes all registered hook handles."""
    for handle in hook_handles:
        handle.remove()
    hook_handles.clear()

def make_forward_hook(name):
    """Factory to create a forward hook storing activation and registering grad hook."""
    def hook_fn(module, input, output):
        # Store activation (handle tuple output format)
        act_tensor = output[0] if isinstance(output, tuple) else output
        activations[name] = act_tensor
        # Register backward hook on the activation tensor if it requires grad
        if act_tensor.requires_grad:
            # Use lambda with default arg to capture current 'name' value correctly
            act_tensor.register_hook(lambda grad, name=name: gradients.update({name: grad}))
    return hook_fn

def register_model_hooks(model):
    """Registers forward hooks on embeddings and standard layer components."""
    clear_hooks_data() # Clear previous data first
    remove_all_hooks() # Remove any lingering hooks
    print("Registering hooks...")
    hook_added = False

    # --- Hook Embeddings ---
    embed_module = None
    embed_name = "embeddings" # Generic name
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'): # Llama/Qwen-like
        embed_module = model.model.embed_tokens
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'): # GPT-2 like
        embed_module = model.transformer.wte
    # Add elif for other architectures as needed

    if embed_module:
        hook_handles.append(embed_module.register_forward_hook(make_forward_hook(embed_name)))
        print(f"  Hooked {embed_name}")
        hook_added = True
    else:
        print("  Warning: Could not find standard embeddings module.")
        # raise RuntimeError("Embeddings module not found, cannot proceed.") # Or just warn

    # --- Hook Transformer Layers ---
    layers_module = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'): # Llama/Qwen-like
        layers_module = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'): # GPT-2 like
        layers_module = model.transformer.h
    # Add elif for other architectures

    if layers_module:
        print(f"  Hooking {len(layers_module)} transformer layers...")
        for i, layer in enumerate(layers_module):
            # Attention Output (adapt submodule name: 'self_attn', 'attn')
            attn_module = getattr(layer, 'self_attn', getattr(layer, 'attn', None))
            if attn_module:
                name = f"L{i}_Attn" # Use simpler name for graph nodes
                hook_handles.append(attn_module.register_forward_hook(make_forward_hook(name)))
                hook_added = True

            # MLP Output (adapt submodule name: 'mlp')
            # TODO: Add more specific MLP checks if needed (e.g., GPT2's mlp.c_proj)
            mlp_module = getattr(layer, 'mlp', None)
            if mlp_module:
                name = f"L{i}_MLP" # Use simpler name for graph nodes
                hook_handles.append(mlp_module.register_forward_hook(make_forward_hook(name)))
                hook_added = True

            if not attn_module and not mlp_module:
                 print(f"    Warning: Layer {i} - No standard Attn or MLP module found to hook.")

    else:
        print("  Warning: Could not find standard transformer layers module ('layers' or 'h').")
        # raise RuntimeError("Transformer layers not found, cannot proceed.") # Or just warn

    if not hook_added:
         raise RuntimeError("Failed to add any hooks to the model.")
    print(f"Registered {len(hook_handles)} hooks successfully.")


# --- Attribution Computation Function ---
def compute_grad_activation_attributions(model, tokenizer, prompt, device, dtype):
    """Computes Grad*Activation norms for hooked components for a given prompt."""
    print("\n--- Computing Attributions ---")
    start_time = time.time()
    model.eval()
    attribution_scores = {}
    predicted_token = "N/A"
    predicted_token_id = -1
    # Initialize variables to ensure they exist even if errors occur before assignment
    inputs, outputs, last_token_logits, target_logit = None, None, None, None

    try:
        # 1. Register Hooks
        register_model_hooks(model)

        # 2. Forward Pass
        print("Running forward pass...")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # <<< MODIFICATION START >>>
        # Conditionally use autocast based on device type support
        if device == 'cuda':
            # Use autocast specifically for CUDA
            print(f"Using torch.autocast for CUDA with dtype: {dtype}")
            with torch.autocast(device_type='cuda', dtype=dtype):
                outputs = model(**inputs)
        else:
            # For MPS or CPU, run without explicit autocast context manager.
            # The model layers should already use the specified TORCH_DTYPE
            # from loading if supported by the operations.
            print(f"Running forward pass directly on {device.upper()} (autocast context manager skipped/unsupported for this device type)")
            outputs = model(**inputs)
        # <<< MODIFICATION END >>>

        # Check if outputs is None (e.g., if model(**inputs) failed silently, though unlikely)
        if outputs is None:
            raise RuntimeError("Model forward pass did not return outputs.")

        logits = outputs.logits
        # Select logits for the last token, ensure float32 for stability
        last_token_logits = logits[0, -1].float()
        predicted_token_id = int(last_token_logits.argmax())
        try:
             predicted_token = tokenizer.decode([predicted_token_id])
        except Exception: predicted_token = "[Decode Error]"
        print(f"Forward pass done. Predicted next token: '{predicted_token}' (ID: {predicted_token_id})")

        # 3. Backward Pass (from the chosen logit)
        print("Running backward pass...")
        target_logit = last_token_logits[predicted_token_id]
        model.zero_grad() # Crucial before backward
        target_logit.backward()
        print("Backward pass done.")

        # 4. Calculate Grad * Activation Norms
        print("Calculating Grad*Activation norms...")
        with torch.no_grad():
            for name, act in activations.items():
                grad = gradients.get(name)
                if grad is None:
                    print(f"  Warning: Gradient missing for {name}")
                    continue
                # Ensure devices match if necessary (shouldn't be needed with proper hooks)
                if grad.device != act.device: grad = grad.to(act.device)
                # Calculate norm using float32 for stability
                norm = torch.norm((grad.float() * act.float())).cpu().item()
                attribution_scores[name] = norm

        print(f"Attribution calculation finished ({time.time() - start_time:.2f}s)")

    except Exception as e:
        print(f"\n!!! Error during attribution computation: {e}")
        import traceback
        traceback.print_exc()
        attribution_scores = None # Indicate failure
    finally:
        # 5. Cleanup Hooks and Variables
        print("\nCleaning up attribution resources...")
        remove_all_hooks()
        clear_hooks_data()
        # Delete variables, checking if they were assigned first to avoid UnboundLocalError
        if 'inputs' in locals() and inputs is not None: del inputs
        if 'outputs' in locals() and outputs is not None: del outputs
        if 'last_token_logits' in locals() and last_token_logits is not None: del last_token_logits
        if 'target_logit' in locals() and target_logit is not None: del target_logit
        gc.collect()
        if device == 'cuda': torch.cuda.empty_cache()
        # if device == 'mps': torch.mps.empty_cache() # If using recent torch

    return attribution_scores, predicted_token

# --- Node Sorting Utility (shared) ---
def get_node_sort_key(node_name):
    """
    Sort key for nodes named:
      - "embeddings" or "START"
      - "L{idx}_Attn" or "L{idx}_MLP"
      - "TARGET"
    Ensures embeddings/START come first, then layers in order (Attn before MLP),
    and TARGET last.
    """
    if node_name in ("embeddings", "START"):
        return -1
    if node_name == "TARGET":
        return float('inf')
    # Expecting names like "L{layer}_{Type}"
    try:
        parts = node_name.split('_')
        layer_idx = int(parts[0][1:])  # drop leading 'L'
        comp = parts[1]
        # Attn before MLP within each layer
        return layer_idx * 2 + (0 if comp == "Attn" else 1)
    except Exception:
        # Unrecognized names go at the end but before TARGET
        return float('inf') - 1

# --- Graph Visualization Function ---
def build_and_visualize_attribution_graph(attributions, target_token):
    """Builds and displays a NetworkX graph of the attributions."""
    if not attributions:
        print("No attributions computed, cannot build graph.")
        return

    print("\n--- Building and Visualizing Attribution Graph ---")
    G = nx.DiGraph()

    sorted_nodes = sorted(attributions.keys(), key=get_node_sort_key)
    graph_nodes = ["START"] + sorted_nodes + ["TARGET"] # Add start/end markers

    # Add nodes to graph, storing attribution score as 'weight' attribute
    G.add_node("START", weight=0.0) # Placeholder start
    for node_name in sorted_nodes:
        G.add_node(node_name, weight=attributions.get(node_name, 0.0))
    G.add_node("TARGET", weight=0.0) # Placeholder end representing the chosen logit

    # Add edges sequentially along the path
    # Edge weight = attribution score of the SOURCE node (influence flowing out)
    max_weight = 0.0
    for i in range(len(graph_nodes) - 1):
        src = graph_nodes[i]
        dst = graph_nodes[i+1]
        # Use source node's computed attribution as edge weight
        # For START node, weight is 0
        weight = G.nodes[src]['weight']
        G.add_edge(src, dst, weight=weight)
        max_weight = max(max_weight, weight)


    # Visualization using NetworkX and Matplotlib
    plt.figure(figsize=(max(10, len(graph_nodes) * 0.8), 6)) # Adjust figure size
    pos = nx.spring_layout(G, seed=42, k=0.9) # k adjusts spacing

    # Normalize edge weights for visualization width (add epsilon for stability)
    epsilon = 1e-9
    norm_denominator = max(epsilon, max_weight)
    edge_widths = [G.edges[u, v]['weight'] / norm_denominator * 5 + 0.5 for u, v in G.edges()] # Min width 0.5

    # Get node sizes based on weight (optional, can make graph busy)
    # node_weights = [G.nodes[n]['weight'] for n in G.nodes()]
    # max_node_weight = max(epsilon, max(node_weights))
    # node_sizes = [w / max_node_weight * 1500 + 500 for w in node_weights] # Min size 500
    node_sizes = 1000 # Use fixed size for clarity


    nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color="lightblue",
            font_size=9, arrowsize=15, width=edge_widths, edge_color="gray", connectionstyle="arc3,rad=0.1")

    # Add edge labels (attribution scores)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3)

    plt.title(f"Attribution Graph (Grad*Activation Proxy)\nPathway to Target Token: '{target_token}'")
    plt.tight_layout()
    plt.show()


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Dependency Check ---
    try:
        import networkx
        import matplotlib
    except ImportError:
        print("Error: Missing dependencies. Please install them:")
        print("pip install networkx matplotlib")
        sys.exit(1)

    # --- Load Model and Tokenizer ---
    model = None
    tokenizer = None
    try:
        print(f"\nLoading tokenizer '{MODEL_NAME}'...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set tokenizer pad_token to eos_token.")

        print(f"Loading model '{MODEL_NAME}' to {DEVICE.upper()} with {DTYPE_NAME}...")
        load_start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=TORCH_DTYPE,
            trust_remote_code=True,
            low_cpu_mem_usage=(DEVICE != 'cpu') # Optimize loading if using GPU/MPS
        )
        # Move model to device if device_map wasn't used implicitly
        if not hasattr(model, 'hf_device_map'):
            model.to(DEVICE)
        load_end = time.time()
        print(f"Model loaded successfully ({load_end - load_start:.2f}s).")

        # --- Compute Attributions ---
        attribution_scores, predicted_token = compute_grad_activation_attributions(
            model, tokenizer, PROMPT, DEVICE, TORCH_DTYPE
        )

        # --- Print Scores and Visualize Graph ---
        if attribution_scores:
            print("\n--- Computed Attribution Scores (Grad*Activation Norms) ---")
            # Sort scores for printing
            sorted_keys = sorted(attribution_scores.keys(), key=get_node_sort_key) # Use same key as graph
            for name in sorted_keys:
                print(f"  {name:<20}: {attribution_scores[name]:.4f}")

            # Visualize
            build_and_visualize_attribution_graph(attribution_scores, predicted_token)
        else:
            print("\nAttribution computation failed. Cannot visualize graph.")

    except ImportError as e: # Catch potential missing libraries again
         print(f"\nError: Missing library. {e}")
         print("Please ensure 'torch', 'transformers', 'networkx', 'matplotlib' are installed.")
    except Exception as e:
        print(f"\nAn unexpected error occurred in the main block: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Final Cleanup ---
        print("\nCleaning up resources...")
        remove_all_hooks() # Ensure hooks are removed
        clear_hooks_data()
        del model # Explicitly delete model and tokenizer
        del tokenizer
        gc.collect() # Force garbage collection
        if DEVICE == 'cuda': torch.cuda.empty_cache()
        # if DEVICE == 'mps': torch.mps.empty_cache() # If using recent torch
        print("Cleanup finished.")

    print("\n--- Prototype Execution Finished ---")