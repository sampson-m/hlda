#!/usr/bin/env python3
"""
Unified configuration system for simulation datasets.

This module defines all simulation configurations in a clean, maintainable way,
replacing scattered shell scripts with a single source of truth.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import yaml


@dataclass
class SimulationConfig:
    """Configuration for a simulation dataset."""
    name: str
    identities: List[str]
    activity_topics: List[str]
    dirichlet_params: List[float]
    activity_fraction: float
    cells_per_identity: int
    n_genes: int
    n_de_genes: int
    de_sigma: float
    de_means: List[float]
    seed: int = 42


@dataclass
class PipelineConfig:
    """Configuration for running model fitting pipelines."""
    dataset_name: str
    data_root: str
    estimates_root: str
    topic_config: str
    n_identity_topics: int
    n_extra_topics: int
    n_total_topics: int
    de_means: List[str] = None  # If None, use all available


# Define all simulation configurations
SIMULATION_CONFIGS = {
    "AB_V1": SimulationConfig(
        name="AB_V1",
        identities=["A", "B"],
        activity_topics=["V1"],
        dirichlet_params=[8, 8, 8],
        activity_fraction=0.3,
        cells_per_identity=3000,
        n_genes=2000,
        n_de_genes=250,
        de_sigma=0.4,
        de_means=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ),
    
    "ABCD_V1V2": SimulationConfig(
        name="ABCD_V1V2",
        identities=["A", "B", "C", "D"],
        activity_topics=["V1", "V2"],
        dirichlet_params=[8, 8, 8],
        activity_fraction=0.3,
        cells_per_identity=3000,
        n_genes=2000,
        n_de_genes=250,
        de_sigma=0.4,
        de_means=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    ),
    
    "ABCD_V1V2_new": SimulationConfig(
        name="ABCD_V1V2_new",
        identities=["A", "B", "C", "D"],
        activity_topics=["V1", "V2"],
        dirichlet_params=[8, 2, 2],  # Different Dirichlet prior
        activity_fraction=0.3,
        cells_per_identity=3000,
        n_genes=2000,
        n_de_genes=400,  # More DE genes
        de_sigma=0.4,
        de_means=[0.1, 0.2, 0.3, 0.4, 0.5]  # Subset of DE means
    )
}

# Define pipeline configurations
PIPELINE_CONFIGS = {
    "AB_V1": PipelineConfig(
        dataset_name="AB_V1",
        data_root="data/AB_V1",
        estimates_root="estimates/AB_V1",
        topic_config="3_topic_fit",
        n_identity_topics=2,
        n_extra_topics=1,
        n_total_topics=3
    ),
    
    "ABCD_V1V2": PipelineConfig(
        dataset_name="ABCD_V1V2",
        data_root="data/ABCD_V1V2",
        estimates_root="estimates/ABCD_V1V2",
        topic_config="6_topic_fit",
        n_identity_topics=4,
        n_extra_topics=2,
        n_total_topics=6
    ),
    
    "ABCD_V1V2_new": PipelineConfig(
        dataset_name="ABCD_V1V2_new",
        data_root="data/ABCD_V1V2_new",
        estimates_root="estimates/ABCD_V1V2_new",
        topic_config="6_topic_fit",
        n_identity_topics=4,
        n_extra_topics=2,
        n_total_topics=6,
        de_means=["DE_mean_0.1", "DE_mean_0.2", "DE_mean_0.3", "DE_mean_0.4", "DE_mean_0.5"]
    )
}


def get_simulation_config(dataset_name: str) -> SimulationConfig:
    """Get simulation configuration for a dataset."""
    if dataset_name not in SIMULATION_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(SIMULATION_CONFIGS.keys())}")
    return SIMULATION_CONFIGS[dataset_name]


def get_pipeline_config(dataset_name: str) -> PipelineConfig:
    """Get pipeline configuration for a dataset."""
    if dataset_name not in PIPELINE_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(PIPELINE_CONFIGS.keys())}")
    return PIPELINE_CONFIGS[dataset_name]


def list_available_datasets() -> List[str]:
    """List all available dataset names."""
    return list(SIMULATION_CONFIGS.keys())


def validate_configs():
    """Validate that all configurations are consistent."""
    for dataset_name in SIMULATION_CONFIGS:
        if dataset_name not in PIPELINE_CONFIGS:
            raise ValueError(f"Missing pipeline config for {dataset_name}")
        
        sim_config = SIMULATION_CONFIGS[dataset_name]
        pipe_config = PIPELINE_CONFIGS[dataset_name]
        
        # Validate topic counts
        expected_identity_topics = len(sim_config.identities)
        if pipe_config.n_identity_topics != expected_identity_topics:
            raise ValueError(f"Config mismatch for {dataset_name}: expected {expected_identity_topics} identity topics, got {pipe_config.n_identity_topics}")
        
        expected_total_topics = expected_identity_topics + len(sim_config.activity_topics)
        if pipe_config.n_total_topics != expected_total_topics:
            raise ValueError(f"Config mismatch for {dataset_name}: expected {expected_total_topics} total topics, got {pipe_config.n_total_topics}")
    
    print("✓ All configurations validated successfully!")


if __name__ == "__main__":
    # Validate configurations when run directly
    validate_configs()
    
    # Print available configurations
    print("\nAvailable datasets:")
    for name in list_available_datasets():
        sim_config = get_simulation_config(name)
        pipe_config = get_pipeline_config(name)
        print(f"  {name}:")
        print(f"    Identities: {sim_config.identities}")
        print(f"    Activity topics: {sim_config.activity_topics}")
        print(f"    DE means: {sim_config.de_means}")
        print(f"    Total topics: {pipe_config.n_total_topics}")
        print() 