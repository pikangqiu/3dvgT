"""Adapter template for wiring G3T/VGGT-style heads into the project trainer.

This module is intentionally importable through `runtime.model.adapter_module_path`.
It keeps the same prediction contract as the scaffold model while leaving a
focused replacement point for a real G3T/VGGT backbone.
"""

from __future__ import annotations


def build_model(
    *,
    point_count: int = 128,
    bev_channels: int = 8,
    satellite_channels: int = 3,
    latent_dim: int = 128,
):
    """Build a trainable adapter that satisfies the reconstruction contract."""

    import torch
    from torch import nn

    from vggt_project.models.scaffold import SatelliteBEVG3TScaffold

    class G3TVGGTAdapterTemplate(nn.Module):
        """Thin trainable bridge until concrete G3T/VGGT heads are connected."""

        def __init__(self) -> None:
            super().__init__()
            self.backbone = SatelliteBEVG3TScaffold.build(
                bev_channels=bev_channels,
                satellite_channels=satellite_channels,
                latent_dim=latent_dim,
                point_count=point_count,
            )

        def forward(self, batch: dict) -> dict:
            return self.backbone(batch)

        def freeze_backbone(self) -> None:
            # Keep camera-specific prediction heads trainable for adapter smoke tests.
            for name, parameter in self.backbone.named_parameters():
                if name.startswith(("bev_encoder.", "satellite_encoder.", "camera_encoder.", "fusion.")):
                    parameter.requires_grad = False

        @property
        def adapter_status(self) -> dict[str, str]:
            return {
                "status": "template",
                "backbone": "SatelliteBEVG3TScaffold",
                "next_step": "replace backbone/head calls with concrete G3T/VGGT modules",
            }

    return G3TVGGTAdapterTemplate()
