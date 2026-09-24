from .config import Config, ATTRIBUTE_NAMES

__all__ = ["Config", "ATTRIBUTE_NAMES", "VAPORPAR"]


def __getattr__(name):
    # Lazy import keeps lightweight utilities (dataset/checkpoint inspection) usable
    # before the Transformers dependency is imported.
    if name == "VAPORPAR":
        from .model import VAPORPAR
        return VAPORPAR
    raise AttributeError(name)
