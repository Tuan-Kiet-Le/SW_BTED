"""
Fix beta_param handling in SWCostModel when passed as dict or float
"""
import os, sys, yaml
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import apted
import numpy as np
from typing import List, Dict, Any, Optional

from src.node import CapstoneNode

config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {
        "prefilter_threshold": 0.25,
        "max_edit_budget_ratio": 0.4,
        "beta": {"T2": 0.0, "T3": 0.9, "T4": 1.0},
        "w_del": {"T2": 2.0, "T3": 1.0},
        "w_ins": {"T2": 2.0, "T3": 1.0}
    }

DOMAIN_SCHEMA_DIST = {
    ("D1_BUSINESS_CONTEXT", "D1_BUSINESS_CONTEXT"): 0.0,
    ("D2_FUNCTIONAL", "D2_FUNCTIONAL"): 0.0,
    ("D3_TECHNICAL_REALIZATION", "D3_TECHNICAL_REALIZATION"): 0.0,
    ("D4_EXECUTION_PLANNING", "D4_EXECUTION_PLANNING"): 0.0,
    ("D1_BUSINESS_CONTEXT", "D2_FUNCTIONAL"): 0.8,
    ("D2_FUNCTIONAL", "D1_BUSINESS_CONTEXT"): 0.8,
    ("D1_BUSINESS_CONTEXT", "D3_TECHNICAL_REALIZATION"): 0.9,
    ("D3_TECHNICAL_REALIZATION", "D1_BUSINESS_CONTEXT"): 0.9,
    ("D1_BUSINESS_CONTEXT", "D4_EXECUTION_PLANNING"): 0.9,
    ("D4_EXECUTION_PLANNING", "D1_BUSINESS_CONTEXT"): 0.9,
    ("D2_FUNCTIONAL", "D3_TECHNICAL_REALIZATION"): 0.5,
    ("D3_TECHNICAL_REALIZATION", "D2_FUNCTIONAL"): 0.5,
    ("D2_FUNCTIONAL", "D4_EXECUTION_PLANNING"): 0.7,
    ("D4_EXECUTION_PLANNING", "D2_FUNCTIONAL"): 0.7,
    ("D3_TECHNICAL_REALIZATION", "D4_EXECUTION_PLANNING"): 0.6,
    ("D4_EXECUTION_PLANNING", "D3_TECHNICAL_REALIZATION"): 0.6,
}

def cosine_sim(vec_a, vec_b) -> float:
    if vec_a is None or vec_b is None:
        return 0.0
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    val = np.dot(a, b) / (norm_a * norm_b)
    return float(np.clip(val, -1.0, 1.0))

class SWCostModel:
    def __init__(self, alpha=None, beta=None, gamma=None, cso_graph=None, max_depth=19):
        self.cso_graph = cso_graph.to_undirected() if cso_graph else None
        self.max_depth = max_depth
        self.path_length_cache = {}
        
        self.beta = CONFIG["beta"]
        self.w_del_config = CONFIG["w_del"]
        self.w_ins_config = CONFIG["w_ins"]
        self.prefilter_threshold = CONFIG["prefilter_threshold"]
        self.max_edit_budget_ratio = CONFIG["max_edit_budget_ratio"]
        
        self.alpha = alpha if alpha is not None else CONFIG.get("alpha", 0.8)
        self.beta_param = beta

    def w_del(self, u: CapstoneNode) -> float:
        if u.depth == 1:
            return 0.0
        if u.depth == 4:
            weight = getattr(u, 'tfidf_weight', None)
            if weight is None:
                weight = 0.5
            return 0.5 * weight
        layer = f"T{u.depth}"
        return self.w_del_config.get(layer, 1.0)

    def w_ins(self, v: CapstoneNode) -> float:
        if v.depth == 1:
            return 0.0
        if v.depth == 4:
            weight = getattr(v, 'tfidf_weight', None)
            if weight is None:
                weight = 0.5
            return 0.5 * weight
        layer = f"T{v.depth}"
        return self.w_ins_config.get(layer, 1.0)

    def dist_content(self, u: CapstoneNode, v: CapstoneNode) -> float:
        if u.depth != v.depth:
            return 1.0
            
        l = u.depth
        if l == 2:  # Domain
            return 0.0 if u.label == v.label else 1.0
        elif l == 3:  # IntentMatching
            return 1.0 - cosine_sim(u.embedding, v.embedding)
        elif l == 4:  # TerminologyVerification
            return 0.0 if u.label == v.label else 1.0
        return 1.0

    def dist_schema(self, u: CapstoneNode, v: CapstoneNode) -> float:
        if u.depth != v.depth:
            return 1.0
            
        l = u.depth
        if l == 2:  # Domain
            u_lbl = u.label
            v_lbl = v.label
            return DOMAIN_SCHEMA_DIST.get((u_lbl, v_lbl), 1.0)
        else:
            return 0.0 if u.schema_class == v.schema_class else 1.0

    def w_rep(self, u: CapstoneNode, v: CapstoneNode) -> float:
        if u.depth != v.depth:
            return self.w_del(u) + self.w_ins(v)
            
        l = u.depth
        if l == 1:
            return 0.0
            
        layer = f"T{l}"
        if self.beta_param is not None and l not in (2,):
            if isinstance(self.beta_param, dict):
                beta_l = self.beta_param.get(layer, 0.5)
            else:
                beta_l = float(self.beta_param)
        else:
            beta_l = self.beta.get(layer, 0.5)
            
        content_d = self.dist_content(u, v)
        schema_d = self.dist_schema(u, v)
        
        return (self.w_del(u) + self.w_ins(v)) * (beta_l * content_d + (1.0 - beta_l) * schema_d)

def iter_nodes(node: CapstoneNode):
    yield node
    for child in node.children:
        yield from iter_nodes(child)

def sw_bted(tree_a: CapstoneNode, tree_b: CapstoneNode, cost_model: SWCostModel, k: Optional[float] = None) -> dict:
    self_a = sum(cost_model.w_del(n) for n in iter_nodes(tree_a))
    self_b = sum(cost_model.w_ins(n) for n in iter_nodes(tree_b))
    max_possible_cost = self_a + self_b
    
    if k is None:
        k = cost_model.max_edit_budget_ratio * max_possible_cost
        
    w_rep_cache = {}
    w_del_cache = {}
    w_ins_cache = {}
    
    def cached_rename(u, v):
        key = (id(u), id(v))
        if key in w_rep_cache:
            return w_rep_cache[key]
        val = cost_model.w_rep(u, v)
        w_rep_cache[key] = val
        return val
        
    def cached_delete(u):
        uid = id(u)
        if uid in w_del_cache:
            return w_del_cache[uid]
        val = cost_model.w_del(u)
        w_del_cache[uid] = val
        return val
        
    def cached_insert(v):
        vid = id(v)
        if vid in w_ins_cache:
            return w_ins_cache[vid]
        val = cost_model.w_ins(v)
        w_ins_cache[vid] = val
        return val

    config = apted.Config()
    config.rename = cached_rename
    config.delete = cached_delete
    config.insert = cached_insert

    dict_a = {c.schema_class: c for c in tree_a.children}
    dict_b = {c.schema_class: c for c in tree_b.children}
    
    domains = ["D1_BUSINESS_CONTEXT", "D2_FUNCTIONAL", "D3_TECHNICAL_REALIZATION", "D4_EXECUTION_PLANNING"]
    breakdown = {}
    total_dist = 0.0
    pruned = False
    
    for domain in domains:
        child_a = dict_a.get(domain)
        child_b = dict_b.get(domain)
        
        if pruned:
            breakdown[domain] = 0.0
            continue
            
        if child_a and child_b:
            sec_dist = apted.APTED(child_a, child_b, config).compute_edit_distance()
            total_dist += sec_dist
            denom = sum(cost_model.w_del(n) for n in iter_nodes(child_a)) + sum(cost_model.w_ins(n) for n in iter_nodes(child_b))
            sim = 1.0 - sec_dist / denom if denom > 0 else 1.0
            breakdown[domain] = round(sim, 4)
        elif child_a:
            sec_dist = sum(cost_model.w_del(n) for n in iter_nodes(child_a))
            total_dist += sec_dist
            breakdown[domain] = 0.0
        elif child_b:
            sec_dist = sum(cost_model.w_ins(n) for n in iter_nodes(child_b))
            total_dist += sec_dist
            breakdown[domain] = 0.0
        else:
            breakdown[domain] = 1.0
            
        if total_dist > k:
            pruned = True
            total_dist = float('inf')
            
    return {
        "distance": total_dist,
        "breakdown": breakdown,
        "pruned": pruned,
        "max_possible_cost": max_possible_cost
    }

def normalize_similarity(tree_a: CapstoneNode, tree_b: CapstoneNode, cost_model: SWCostModel) -> float:
    sim_global = 0.0
    has_embeddings = False
    if hasattr(tree_a, 'embedding') and hasattr(tree_b, 'embedding') and tree_a.embedding and tree_b.embedding:
        sim_global = cosine_sim(tree_a.embedding, tree_b.embedding)
        has_embeddings = True
        
    if has_embeddings and sim_global < cost_model.prefilter_threshold:
        sim_struct = 0.0
    else:
        res = sw_bted(tree_a, tree_b, cost_model)
        if res.get("pruned") or res["distance"] == float('inf'):
            sim_struct = 0.0
        else:
            denom = res["max_possible_cost"]
            if denom == 0:
                sim_struct = 1.0
            else:
                normalized_cost = res["distance"] / denom
                if normalized_cost > cost_model.max_edit_budget_ratio:
                    sim_struct = 0.0
                else:
                    sim_struct = 1.0 - (normalized_cost / cost_model.max_edit_budget_ratio)
    
    if has_embeddings:
        sim = cost_model.alpha * sim_struct + (1.0 - cost_model.alpha) * sim_global
    else:
        sim = sim_struct
        
    return round(sim, 4)

def dict_to_node(d: dict) -> CapstoneNode:
    return CapstoneNode.from_dict(d)
