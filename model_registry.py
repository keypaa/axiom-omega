"""
Model Registry: Auto-detect model architecture and provide model-specific defaults.

Handles:
- Layer structure (where to hook)
- Hidden dimensions
- Target layer selection defaults
- Architecture-specific parameters
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import re


@dataclass
class ModelInfo:
    """Metadata for a model family"""

    family: str  # "llama", "mistral", "qwen"
    layer_attr: str  # path to layers list (e.g., "model.layers")
    hook_point: str  # "residual_out" or "attention_out"
    default_target_layers: List[int]  # Sensible defaults
    attention_module_name: str  # e.g., "self_attn"
    ffn_module_name: str  # e.g., "mlp"

    def get_layer_module(self, model, layer_idx: int):
        """Get the actual layer module from model"""
        layers = model
        for attr in self.layer_attr.split("."):
            layers = getattr(layers, attr)
        return layers[layer_idx]


class ModelRegistry:
    """Registry of model architectures and their defaults"""

    # Map model identifiers to architecture info
    REGISTRY: Dict[str, ModelInfo] = {}

    @classmethod
    def register(cls, model_id_pattern: str, info: ModelInfo):
        """Register a model architecture"""
        cls.REGISTRY[model_id_pattern] = info

    @classmethod
    def get(cls, model_name: str, config=None) -> ModelInfo:
        """
        Get model info for a model name.

        Args:
            model_name: HF model ID (e.g., "meta-llama/Meta-Llama-3-8B")
            config: Optional model.config for verification

        Returns:
            ModelInfo with architecture details
        """
        # Try exact match first
        if model_name in cls.REGISTRY:
            return cls.REGISTRY[model_name]

        # Try pattern matching
        for pattern, info in cls.REGISTRY.items():
            if cls._pattern_match(model_name, pattern):
                return info

        # Fallback: try to infer from config
        if config is not None:
            return cls._infer_from_config(model_name, config)

        raise ValueError(
            f"Unknown model: {model_name}. Register it first or provide config."
        )

    @classmethod
    def _pattern_match(cls, model_name: str, pattern: str) -> bool:
        """Check if model_name matches pattern (support wildcards)"""
        regex_pattern = pattern.replace("*", ".*")
        return re.fullmatch(regex_pattern, model_name) is not None

    @classmethod
    def _infer_from_config(cls, model_name: str, config) -> ModelInfo:
        """Infer model info from HF config object"""
        # Detect model family from architecture_name or model_type
        arch = getattr(config, "architectures", ["Unknown"])[0].lower()
        model_type = getattr(config, "model_type", "").lower()

        # Try to match
        if "llama" in arch or "llama" in model_type:
            return cls._llama_info(config)
        elif "mistral" in arch or "mistral" in model_type:
            return cls._mistral_info(config)
        elif "qwen" in arch or "qwen" in model_type:
            return cls._qwen_info(config)
        else:
            raise ValueError(
                f"Cannot infer architecture for {model_name} (arch={arch}, type={model_type})"
            )

    @classmethod
    def _llama_info(cls, config) -> ModelInfo:
        """Build ModelInfo for Llama architecture"""
        n_layers = config.num_hidden_layers
        # Select layers: two in middle-to-late thirds
        default_layers = [
            n_layers // 3,  # early-mid boundary
            2 * n_layers // 3,  # mid-late boundary
        ]
        return ModelInfo(
            family="llama",
            layer_attr="model.layers",
            hook_point="residual_out",
            default_target_layers=default_layers,
            attention_module_name="self_attn",
            ffn_module_name="mlp",
        )

    @classmethod
    def _mistral_info(cls, config) -> ModelInfo:
        """Build ModelInfo for Mistral architecture (same as Llama)"""
        return cls._llama_info(config)

    @classmethod
    def _qwen_info(cls, config) -> ModelInfo:
        """Build ModelInfo for Qwen2 architecture"""
        return cls._llama_info(config)


# Register standard models
ModelRegistry.register(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    ModelInfo(
        family="llama",
        layer_attr="model.layers",
        hook_point="residual_out",
        default_target_layers=[10, 31],  # Known good for Llama 3 8B
        attention_module_name="self_attn",
        ffn_module_name="mlp",
    ),
)

ModelRegistry.register(
    "meta-llama/Meta-Llama-3-70B-Instruct",
    ModelInfo(
        family="llama",
        layer_attr="model.layers",
        hook_point="residual_out",
        default_target_layers=[23, 58],  # Proportional to 70B architecture
        attention_module_name="self_attn",
        ffn_module_name="mlp",
    ),
)

ModelRegistry.register(
    "mistralai/Mistral-7B-Instruct-v0.2",
    ModelInfo(
        family="mistral",
        layer_attr="model.layers",
        hook_point="residual_out",
        default_target_layers=[10, 22],  # Estimated for 7B (32 layers)
        attention_module_name="self_attn",
        ffn_module_name="mlp",
    ),
)

ModelRegistry.register(
    "mistralai/*",  # Wildcard for all Mistral variants
    ModelInfo(
        family="mistral",
        layer_attr="model.layers",
        hook_point="residual_out",
        default_target_layers=[],  # Will be auto-selected
        attention_module_name="self_attn",
        ffn_module_name="mlp",
    ),
)

ModelRegistry.register(
    "Qwen/Qwen2-*",  # Wildcard for Qwen2 variants
    ModelInfo(
        family="qwen",
        layer_attr="model.layers",
        hook_point="residual_out",
        default_target_layers=[],  # Will be auto-selected
        attention_module_name="self_attn",
        ffn_module_name="mlp",
    ),
)
