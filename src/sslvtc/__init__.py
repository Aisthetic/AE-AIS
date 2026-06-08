"""SSL-VTC: semi-supervised VAE for vessel trajectory classification (Duan et al. 2022)."""

__version__ = "0.1.0"

CLASSES = ("fishing", "passenger", "cargo", "tanker")
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
