from typing import List, Optional
import numpy as np

class CapstoneNode:
    def __init__(
        self,
        label: str,
        schema_class: str,
        depth: int,
        children: Optional[List['CapstoneNode']] = None,
        feature_label: Optional[str] = None,
        raw_text: Optional[str] = None,
        normalized_text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        cso_ancestors: Optional[List[str]] = None,
        source_role: Optional[str] = None,
        tfidf_weight: Optional[float] = None
    ):
        self.label = label                 # Node label (e.g. Project_ID, domain label, role type, keyword)
        self.schema_class = schema_class   # e.g., D1_BUSINESS_CONTEXT, IntentMatching, TerminologyVerification
        self.depth = depth                 # 1 = ROOT (MacroFilter), 2 = DOMAIN (DomainPartition), 3 = INTENT (IntentMatching), 4 = LEAF/TERM (TerminologyVerification)
        self.children = children if children is not None else []
        self.feature_label = feature_label
        self.raw_text = raw_text
        self.normalized_text = normalized_text
        self.embedding = embedding
        self.cso_ancestors = cso_ancestors if cso_ancestors is not None else []
        self.source_role = source_role
        self.tfidf_weight = tfidf_weight

    def to_dict(self) -> dict:
        d = {
            "label": self.label,
            "schema_class": self.schema_class,
            "depth": self.depth,
            "children": [child.to_dict() for child in self.children]
        }
        if self.feature_label is not None:
            d["feature_label"] = self.feature_label
        if self.raw_text is not None:
            d["raw_text"] = self.raw_text
        if self.normalized_text is not None:
            d["normalized_text"] = self.normalized_text
        if self.embedding is not None:
            d["embedding"] = self.embedding
        if self.cso_ancestors:
            d["cso_ancestors"] = self.cso_ancestors
        if self.source_role is not None:
            d["source_role"] = self.source_role
        if self.tfidf_weight is not None:
            d["tfidf_weight"] = self.tfidf_weight
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'CapstoneNode':
        if not d:
            return None
        node = cls(
            label=d["label"],
            schema_class=d["schema_class"],
            depth=d["depth"],
            feature_label=d.get("feature_label"),
            raw_text=d.get("raw_text"),
            normalized_text=d.get("normalized_text"),
            embedding=d.get("embedding"),
            cso_ancestors=d.get("cso_ancestors"),
            source_role=d.get("source_role"),
            tfidf_weight=d.get("tfidf_weight")
        )
        node.children = [cls.from_dict(child) for child in d.get("children", [])]
        return node
