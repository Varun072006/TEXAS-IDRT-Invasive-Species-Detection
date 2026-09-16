"""
Azure-compatible configuration loader.

The production system ingests drone imagery from a private Azure Blob
Storage container and (in progress) will push trained model artifacts to
Azure ML / Azure Container Instances for scalable inference. None of
those credentials, container names, or endpoints are included here --
this module only defines the *shape* of that configuration, sourced
entirely from environment variables at runtime.

To try this against your own Azure account, copy `.env.example` (create
one locally, it is git-ignored) with your own values -- never commit
real connection strings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AzureBlobConfig:
    """Connection parameters for Azure Blob Storage, read from the environment.

    All fields default to None so the demo pipeline runs fully offline with
    synthetic data when no Azure environment is configured.
    """

    account_url: str | None = None
    container_name: str | None = None
    sas_token: str | None = None

    @classmethod
    def from_env(cls) -> "AzureBlobConfig":
        return cls(
            account_url=os.getenv("AZURE_STORAGE_ACCOUNT_URL"),
            container_name=os.getenv("AZURE_STORAGE_CONTAINER"),
            sas_token=os.getenv("AZURE_STORAGE_SAS_TOKEN"),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.account_url and self.container_name and self.sas_token)


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration. Non-secret, safe defaults only."""

    tile_size_px: int = 1024
    tile_overlap_px: int = 128
    sam_model_type: str = "vit_b"  # swap for vit_h in production for higher fidelity
    embedding_model: str = "dinov2_vitb14"
    cluster_algorithm: str = "hdbscan"
    duplicate_iou_threshold: float = 0.35

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            tile_size_px=int(os.getenv("TILE_SIZE_PX", cls.tile_size_px)),
            tile_overlap_px=int(os.getenv("TILE_OVERLAP_PX", cls.tile_overlap_px)),
            sam_model_type=os.getenv("SAM_MODEL_TYPE", cls.sam_model_type),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            cluster_algorithm=os.getenv("CLUSTER_ALGORITHM", cls.cluster_algorithm),
            duplicate_iou_threshold=float(
                os.getenv("DUPLICATE_IOU_THRESHOLD", cls.duplicate_iou_threshold)
            ),
        )
