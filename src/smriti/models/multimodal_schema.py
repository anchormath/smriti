"""
Pydantic schemas and metadata models for Multimodal Enterprise Artifacts.
Supports architectural diagrams (images), audio design walkthroughs, text specs, and code.
Aligned with Naksha's 4-domain ontology (Sangh, Kala, Krama, Vyapar) and Smriti's MRL hierarchy.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field, field_validator

from smriti.models.project_schema import MRLEmbeddings


class ModalityType(str, Enum):
    TEXT = "text"
    IMAGE_DIAGRAM = "image_diagram"
    AUDIO_WALKTHROUGH = "audio_walkthrough"
    CODE_SPEC = "code_spec"


class EnterpriseDomain(str, Enum):
    SANGH = "Sangh"    # People, Teams, Org
    KALA = "Kala"      # Skills, Career, Reviews
    KRAMA = "Krama"    # Systems, Architecture, Tech Stack, Services
    VYAPAR = "Vyapar"  # Business, Products, Capabilities, Features


class SpatialRegion(BaseModel):
    """Normalized bounding box for diagram elements [ymin, xmin, ymax, xmax] in [0.0, 1.0]."""
    ymin: float = Field(..., ge=0.0, le=1.0)
    xmin: float = Field(..., ge=0.0, le=1.0)
    ymax: float = Field(..., ge=0.0, le=1.0)
    xmax: float = Field(..., ge=0.0, le=1.0)
    component_name: Optional[str] = Field(default=None, description="Name of the labeled component in diagram")


class TemporalSegment(BaseModel):
    """Audio segment time bounds in seconds."""
    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., ge=0.0)
    transcript_snippet: Optional[str] = Field(default=None, description="Spoken transcript snippet for this segment")


class MultimodalArtifact(BaseModel):
    """
    Representation of a multimodal architectural or organizational artifact.
    Can be an architectural diagram, an audio design walkthrough / review, or a technical spec.
    """
    id: str = Field(..., description="Unique artifact identifier, e.g. art_krama_diagram_event_gw")
    domain: EnterpriseDomain = Field(..., description="Top-level enterprise domain (Sangh, Kala, Krama, Vyapar)")
    subdomain: str = Field(..., description="Specialized subdomain, e.g. service, data_store, review, product_feature")
    parent_entity_id: str = Field(..., description="Linked entity ID in EKG, e.g. service:event_gateway")
    modality: ModalityType = Field(..., description="Modality type (text, image_diagram, audio_walkthrough, code_spec)")
    
    title: str = Field(..., description="Descriptive title of artifact")
    description: str = Field(..., description="Summary of artifact contents and purpose")
    media_uri: Optional[str] = Field(default=None, description="File path, URL, or URI to media asset")
    
    # Fine-grained anchors
    spatial_region: Optional[SpatialRegion] = Field(default=None, description="Spatial anchor in a diagram")
    temporal_segment: Optional[TemporalSegment] = Field(default=None, description="Temporal anchor in an audio track")
    
    # Ground-truth linkages for cross-modal validation
    linked_artifact_ids: List[str] = Field(default_factory=list, description="IDs of co-occurring or related artifacts")
    tags: List[str] = Field(default_factory=list, description="Architectural and semantic tags")
    
    # Multi-resolution Matryoshka embeddings
    embeddings: Optional[MRLEmbeddings] = Field(default=None, description="MRL unit-normalized embeddings")

    def to_unified_text(self) -> str:
        """Serializes the artifact to standard MRL text representation."""
        anchor_info = ""
        if self.spatial_region and self.spatial_region.component_name:
            anchor_info = f" [Component: {self.spatial_region.component_name}]"
        elif self.temporal_segment and self.temporal_segment.transcript_snippet:
            anchor_info = f" [Spoken: {self.temporal_segment.transcript_snippet}]"

        return (
            f"search_document: [Domain: {self.domain.value}] [Sub-domain: {self.subdomain}] [Modality: {self.modality.value}]\n"
            f"Title: {self.title}\n"
            f"Description: {self.description}{anchor_info}\n"
            f"Tags: {', '.join(self.tags)}"
        )

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flattens artifact for columnar/tabular export."""
        return {
            "id": self.id,
            "domain": self.domain.value,
            "subdomain": self.subdomain,
            "parent_entity_id": self.parent_entity_id,
            "modality": self.modality.value,
            "title": self.title,
            "description": self.description,
            "media_uri": self.media_uri,
            "tags": self.tags,
            "linked_artifact_ids": self.linked_artifact_ids,
            "unified_text": self.to_unified_text(),
            "mrl_d0_64": self.embeddings.d0_64 if self.embeddings else None,
            "mrl_d1_128": self.embeddings.d1_128 if self.embeddings else None,
            "mrl_d2_768": self.embeddings.d2_768 if self.embeddings else None,
        }
