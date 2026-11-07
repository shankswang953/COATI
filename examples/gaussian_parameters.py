"""Concise public configuration for the fresh Gaussian comparison."""
import coati as ct


def make_flow_config(*, iterations=3000, secondary_weight=.5):
    """Override only settings that differ from the COATI defaults."""
    return ct.Config(
        multimodal=True,
        balanced=True,
        iterations=iterations,
        batch_size=256,
        hidden_dim=256,
        secondary_weight=secondary_weight,
        extra={
            "time_plot": True,
            "median_n_sample": 512,
            "floor_n_sample": 256,
            "sync_density_loss": True,
            "map_type": "MLP",
            "adaptive_lambda": True,
        },
    )
