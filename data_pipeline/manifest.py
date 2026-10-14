import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def compute_file_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class ManifestManager:
    def __init__(self, manifest_path: str = "data/manifest.json"):
        self.manifest_path = manifest_path
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "project": "Vak",
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "sources": [],
            "processing_stages": {},
            "split_statistics": {},
            "tokenization": {},
            "artifacts": {},
        }

    def save(self) -> None:
        self.data["last_updated"] = datetime.now(timezone.utc).isoformat()
        os.makedirs(os.path.dirname(os.path.abspath(self.manifest_path)), exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def record_source(
        self,
        name: str,
        source_url: str,
        license_name: str,
        permitted_uses: str,
        language: str,
        domain: str,
        acquisition_method: str,
        is_synthetic: bool,
        revision: str = "main",
        raw_document_count: int = 0,
    ) -> None:
        source_entry = {
            "name": name,
            "source_url": source_url,
            "revision": revision,
            "license": license_name,
            "permitted_uses": permitted_uses,
            "language": language,
            "domain": domain,
            "acquisition_method": acquisition_method,
            "content_type": "synthetic" if is_synthetic else "human-written",
            "raw_document_count": raw_document_count,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        # Update existing or append new
        for i, existing in enumerate(self.data["sources"]):
            if existing["name"] == name:
                self.data["sources"][i] = source_entry
                self.save()
                return
        self.data["sources"].append(source_entry)
        self.save()

    def record_processing(self, stage_name: str, details: Dict[str, Any]) -> None:
        self.data["processing_stages"][stage_name] = {
            **details,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save()

    def record_splits(self, split_stats: Dict[str, Any]) -> None:
        self.data["split_statistics"] = split_stats
        self.save()

    def record_tokenization(self, token_stats: Dict[str, Any]) -> None:
        self.data["tokenization"] = token_stats
        self.save()

    def record_artifact(self, filepath: str, description: str) -> None:
        if os.path.exists(filepath):
            file_hash = compute_file_sha256(filepath)
            size_bytes = os.path.getsize(filepath)
            self.data["artifacts"][os.path.basename(filepath)] = {
                "path": filepath,
                "description": description,
                "sha256": file_hash,
                "size_bytes": size_bytes,
            }
            self.save()
