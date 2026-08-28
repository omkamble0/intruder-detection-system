"""
GOLIATH Control Plane - Centralized Model Registry & Distribution Manager
Generates cryptographic integrity manifests, manages S3/CDN artifacts, and coordinates canary deployments.
"""

import os
import hashlib
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("goliath.control.model_registry")

class ModelRegistry:
    """
    Centralized model artifact registry with cryptographic verification and rollout management.
    """

    def __init__(self, storage_root: str = "models_storage"):
        self.storage_root = storage_root
        os.makedirs(self.storage_root, exist_ok=True)
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._canary_cohorts: Dict[str, float] = {}  # model_version -> rollout percentage (0.0 - 1.0)

    @staticmethod
    def calculate_sha256(filepath: str) -> str:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Artifact not found: {filepath}")
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def register_model_version(
        self,
        model_name: str,
        version: str,
        artifact_path: str,
        input_shape: List[int],
        confidence_threshold: float,
        iou_threshold: float,
        target_accelerators: List[str],
        base_cdn_url: str = "https://cdn.goliath-sec.internal/models",
    ) -> Dict[str, Any]:
        """
        Calculates SHA-256 integrity hash, creates signed manifest, and indexes model for distribution.
        """
        sha256_hash = self.calculate_sha256(artifact_path)
        file_size_bytes = os.path.getsize(artifact_path)
        artifact_filename = os.path.basename(artifact_path)

        manifest = {
            "model_name": model_name,
            "version": version,
            "artifact_filename": artifact_filename,
            "sha256": sha256_hash,
            "file_size_bytes": file_size_bytes,
            "input_shape": input_shape,
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "supported_accelerators": target_accelerators,
            "artifact_url": f"{base_cdn_url}/{version}/{artifact_filename}",
            "manifest_url": f"{base_cdn_url}/{version}/manifest.json",
        }

        model_key = f"{model_name}:{version}"
        self._registry[model_key] = manifest
        
        # Save manifest locally in storage dir
        version_dir = os.path.join(self.storage_root, version)
        os.makedirs(version_dir, exist_ok=True)
        manifest_path = os.path.join(version_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            f"[ModelRegistry] Registered model={model_key} sha256={sha256_hash[:12]} size={file_size_bytes}B"
        )
        return manifest

    def get_manifest(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        return self._registry.get(f"{model_name}:{version}")

    def set_canary_rollout(self, version: str, percentage: float):
        """Sets rollout percentage: e.g. 0.05 (5%), 0.25 (25%), 1.0 (100%)."""
        percentage = max(0.0, min(1.0, percentage))
        self._canary_cohorts[version] = percentage
        logger.info(f"[ModelRegistry] Set canary rollout for version {version} to {percentage * 100:.1f}%")

    def assign_model_for_node(self, node_id: str, hardware_arch: str, accelerator: str) -> Optional[Dict[str, Any]]:
        """
        Determines target model version based on hardware compatibility and canary hash bucket.
        """
        # Node hash bucketing for deterministic canary distribution
        node_hash = int(hashlib.md5(node_id.encode()).hexdigest(), 16) % 100
        normalized_bucket = node_hash / 100.0

        for version, rollout_pct in sorted(self._canary_cohorts.items(), key=lambda x: x[1]):
            if normalized_bucket < rollout_pct:
                for key, manifest in self._registry.items():
                    if manifest["version"] == version:
                        if accelerator in manifest["supported_accelerators"] or "cpu" in manifest["supported_accelerators"]:
                            return manifest
        
        # Default to latest registered version if no canary matched
        if self._registry:
            return list(self._registry.values())[-1]
        return None
