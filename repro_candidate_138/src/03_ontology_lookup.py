import os
import pickle
import xml.etree.ElementTree as ET
import networkx as nx
from typing import List, Dict, Tuple, Optional
import difflib

class CSOLookup:
    def __init__(self, owl_path: str = r"Data\cso_v3.5\CSO.3.5.owl", cache_path: str = r"data\processed\cso_graph.pkl"):
        self.owl_path = owl_path
        self.cache_path = cache_path
        self.graph = nx.DiGraph()
        self.concept_to_label: Dict[str, str] = {}
        self.label_to_concept: Dict[str, str] = {}
        self.max_depth = 1
        self.node_depths: Dict[str, int] = {}
        
        self.load_ontology()
        
    def _uri_to_key(self, uri: str) -> str:
        # Extract the topic key from URI
        # e.g., https://cso.kmi.open.ac.uk/topics/computer_science -> computer science
        if "/topics/" in uri:
            key = uri.split("/topics/")[-1]
        else:
            key = uri.split("/")[-1]
        
        # Decode URL encoding like %20, %2C
        import urllib.parse
        key = urllib.parse.unquote(key)
        
        # Replace underscores and hyphens with space
        key = key.replace("_", " ").replace("-", " ").lower().strip()
        return key

    def load_ontology(self):
        if os.path.exists(self.cache_path):
            try:
                print(f"Loading CSO Graph from cache: {self.cache_path}...")
                with open(self.cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.graph = data['graph']
                    self.concept_to_label = data['concept_to_label']
                    self.label_to_concept = data['label_to_concept']
                    self.max_depth = data['max_depth']
                    self.node_depths = data.get('node_depths', {})
                print(f"CSO Graph loaded from cache with {self.graph.number_of_nodes()} nodes.")
                return
            except Exception as e:
                print(f"Error loading cache, rebuilding graph: {e}")

        print(f"Parsing CSO Ontology from {self.owl_path}...")
        if not os.path.exists(self.owl_path):
            print(f"Warning: OWL file not found at {self.owl_path}. Using empty CSO Graph.")
            return

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        
        # Parse XML
        tree = ET.parse(self.owl_path)
        root = tree.getroot()
        
        ns_rdf = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
        ns_rdfs = "{http://www.w3.org/2000/01/rdf-schema#}"
        ns_cso = "{http://cso.kmi.open.ac.uk/schema/cso#}"
        
        for desc in root.findall(f"{ns_rdf}Description"):
            about = desc.attrib.get(f"{ns_rdf}about")
            if not about:
                continue
                
            subject_key = self._uri_to_key(about)
            
            # Find label
            label_node = desc.find(f"{ns_rdfs}label")
            label_text = label_node.text.strip().lower() if label_node is not None and label_node.text else subject_key
            
            self.concept_to_label[subject_key] = label_text
            self.label_to_concept[label_text] = subject_key
            self.graph.add_node(subject_key, label=label_text)
            
            # Parse relationships
            for child in desc:
                if child.tag == f"{ns_cso}superTopicOf":
                    resource = child.attrib.get(f"{ns_rdf}resource")
                    if resource:
                        child_key = self._uri_to_key(resource)
                        self.graph.add_edge(subject_key, child_key)
        
        # Compute max depth and precompute individual node depths using BFS (robust to cycles)
        self.node_depths = {}
        roots = [n for n, d in self.graph.in_degree() if d == 0]
        if not roots:
            roots = list(self.graph.nodes())[:10]  # fallback if no root
            
        import collections
        queue = collections.deque([(root, 1) for root in roots])
        for root in roots:
            self.node_depths[root] = 1
            
        while queue:
            node, curr_depth = queue.popleft()
            for child in self.graph.successors(node):
                if child not in self.node_depths:
                    self.node_depths[child] = curr_depth + 1
                    queue.append((child, curr_depth + 1))
                    
        self.max_depth = max(self.node_depths.values()) if self.node_depths else 10
        print(f"Ontology loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges. Max depth: {self.max_depth}")
        
        # Save to cache
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump({
                    'graph': self.graph,
                    'concept_to_label': self.concept_to_label,
                    'label_to_concept': self.label_to_concept,
                    'max_depth': self.max_depth,
                    'node_depths': self.node_depths
                }, f)
            print(f"CSO Graph saved to cache: {self.cache_path}")
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def lookup(self, label: str) -> Optional[dict]:
        """
        Tra cứu khái niệm trong CSO. 
        Trả về dict chứa canonical concept, depth và ancestors.
        """
        label_clean = label.strip().lower()
        concept = None
        
        # 1. Exact match label or concept key
        if label_clean in self.label_to_concept:
            concept = self.label_to_concept[label_clean]
        elif label_clean in self.concept_to_label:
            concept = label_clean
            
        # 2. Fuzzy match fallback
        if not concept:
            matches = difflib.get_close_matches(label_clean, self.label_to_concept.keys(), n=1, cutoff=0.8)
            if matches:
                concept = self.label_to_concept[matches[0]]
                
        if not concept or concept not in self.graph:
            return None
            
        # Find ancestors (nodes that can reach this node in the directed graph)
        # Note: nx.ancestors returns all nodes that have a path to `concept` (i.e. supertopics)
        super_topics = list(nx.ancestors(self.graph, concept))
        
        # Calculate depth using precomputed topological depths
        depth = self.node_depths.get(concept, 1)
                
        return {
            "cso_concept": concept,
            "label": self.concept_to_label[concept],
            "ancestors": super_topics,
            "depth": depth
        }

if __name__ == "__main__":
    cso = CSOLookup()
    print(cso.lookup("Spring Boot"))
