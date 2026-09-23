"""Feature-set specifications and registry."""

from dataclasses import dataclass

from ..data import EC, PH, SS, TEMP, TOC


@dataclass(frozen=True)
class Experiment:
    name: str
    eem: str = "none"
    tabular: tuple = ()


FeatureSpec = Experiment


def experiment_catalog():
    experiments = [
        Experiment("TOC", tabular=(TOC,)),
        Experiment("TOC_SS_EC", tabular=(TOC, SS, EC)),
        Experiment("SS_EC", tabular=(SS, EC)),
        Experiment("Temp", tabular=(TEMP,)),
        Experiment("pH", tabular=(PH,)),
        Experiment("Temp_pH", tabular=(TEMP, PH)),
        Experiment("Temp_SS_EC", tabular=(TEMP, SS, EC)),
        Experiment("pH_SS_EC", tabular=(PH, SS, EC)),
        Experiment("Temp_pH_SS_EC", tabular=(TEMP, PH, SS, EC)),
    ]
    for prefix, eem in [
        ("EEM", "raw"),
        ("EEMpca", "pca"),
    ]:
        for suffix, tab in [
            ("", ()),
            ("_TOC", (TOC,)),
            ("_SS_EC", (SS, EC)),
            ("_TOC_SS_EC", (TOC, SS, EC)),
            ("_Temp", (TEMP,)),
            ("_pH", (PH,)),
            ("_Temp_pH", (TEMP, PH)),
            ("_Temp_SS_EC", (TEMP, SS, EC)),
            ("_pH_SS_EC", (PH, SS, EC)),
            ("_Temp_pH_SS_EC", (TEMP, PH, SS, EC)),
        ]:
            experiments.append(Experiment(prefix + suffix, eem, tab))
    return {exp.name: exp for exp in experiments}


def feature_catalog():
    """Return the classical feature-set catalog."""
    return experiment_catalog()

__all__ = ["Experiment", "FeatureSpec", "experiment_catalog", "feature_catalog"]
