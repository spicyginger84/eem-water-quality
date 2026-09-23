"""Registry of model names exposed by the classical CLI."""

CLASSICAL_MODELS = ("linear", "tree", "xgboost", "svr")


def available_models():
    """Return model names in stable CLI order."""
    return list(CLASSICAL_MODELS)

