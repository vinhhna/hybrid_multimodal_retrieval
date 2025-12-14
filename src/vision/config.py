"""
Configuration dataclass for Phase 5 vision-grounded detection.

Provides VisionConfig to parse the phase5 section from entity_graph.yaml.

Phase 5 Day 1: Scaffold config parsing for use in later days.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class VisionConfig:
    """
    Configuration for Phase 5 vision-grounded detection and grounding.
    
    Parsed from the `phase5` section in configs/entity_graph.yaml.
    All paths are relative to the project root unless absolute.
    
    Attributes:
        model_id: Detector model identifier (e.g., "owlvit", "owlvit-large")
        batch_size_images: Number of images to process in parallel
        batch_size_phrases: Number of phrase prompts per detection batch
        cache_dir: Directory for model weights/cache
        raw_shards_dir: Directory for raw detection outputs
        post_shards_dir: Directory for postprocessed detection outputs
        shard_size: Number of images per shard file
        
        vis_prior_topK: Max entities from visual prior to consider
        tau_vis: Min confidence threshold for visual prior filtering
        C_max: Hard cap on total candidate phrases per image
        
        tau_sim: Min cosine similarity for neighbor inclusion
        use_mutual_knn: Require bidirectional k-NN for safer neighbors
        min_df_vis: Min visual document frequency for neighbor eligibility
        require_in_vis_prior: If true, only expand to entities in visual prior
        
        null_phrases: List of nonsense phrases for noise estimation
        noise_percentile: Percentile of null phrase scores to use as noise floor
        delta: Safety margin above noise floor for presence threshold
        
        threshold_strategy: "grouped" or "global"
        threshold_objective: "fixed_fpr" or "precision"
        target_fpr: Target false positive rate (if objective=fixed_fpr)
        target_precision: Target precision (if objective=precision)
        min_support: Min number of positive examples for threshold fitting
        
        binding_enable: Toggle binding verification
        binding_topN_images: Number of top-ranked images to verify
        binding_topB_boxes: Number of top-scored boxes per image
        hsv_ranges: HSV color ranges for attribute verification
        clip_fallback_enable: Use CLIP region similarity if HSV fails
        clip_fallback_template: Prompt template for CLIP fallback
        
        pmi_smoothing_eps: Laplace smoothing for PMI calculation
        pmi_npmi_min: Min NPMI threshold for keeping edges
        pmi_npmi_min_stop: Stricter NPMI threshold for stop-nodes
        pmi_degree_cap: Max degree per node
        pmi_per_node_topk: Keep at most top-k edges per node by NPMI
        pmi_stop_nodes_topM_df: Nodes with DF in top M are stop-nodes
    
    Phase 5 usage:
        - Day 1: Define config schema
        - Day 3+: Use in detector, postprocessor, and candidate generation
    """
    
    # Detector settings
    model_id: str = "owlvit"
    batch_size_images: int = 8
    batch_size_phrases: int = 128
    cache_dir: str = "data/vision/cache"
    raw_shards_dir: str = "data/vision/raw_shards"
    post_shards_dir: str = "data/vision/post_shards"
    shard_size: int = 500
    
    # Candidate generation
    vis_prior_topK: int = 50
    tau_vis: float = 0.20
    C_max: int = 128
    
    # Safe neighbors
    tau_sim: float = 0.25
    use_mutual_knn: bool = True
    min_df_vis: int = 5
    require_in_vis_prior: bool = False
    
    # Negative controls
    null_phrases: List[str] = field(default_factory=lambda: [
        "xyzzy nonsense phrase 1",
        "blark random phrase 2"
    ])
    noise_percentile: float = 95.0
    delta: float = 0.05
    
    # Thresholds
    threshold_strategy: str = "grouped"
    threshold_objective: str = "fixed_fpr"
    target_fpr: float = 0.05
    target_precision: float = 0.90
    min_support: int = 200
    
    # Binding verification
    binding_enable: bool = False
    binding_topN_images: int = 50
    binding_topB_boxes: int = 5
    hsv_ranges: Dict[str, Dict[str, int]] = field(default_factory=dict)
    clip_fallback_enable: bool = True
    clip_fallback_template: str = "a photo of a {attr} {obj}"
    
    # PMI pruning
    pmi_smoothing_eps: float = 1.0
    pmi_npmi_min: float = 0.10
    pmi_npmi_min_stop: float = 0.20
    pmi_degree_cap: int = 500
    pmi_per_node_topk: int = 200
    pmi_stop_nodes_topM_df: int = 50
    
    @classmethod
    def from_yaml_dict(cls, phase5_cfg: Dict[str, Any]) -> VisionConfig:
        """
        Create VisionConfig from the phase5 section of entity_graph.yaml.
        
        This helper extracts and flattens all nested config sections:
          - candidates, safe_neighbors, detector, negative_controls,
            thresholds, binding, PMI
        
        Args:
            phase5_cfg: The "phase5" section dict from the YAML config.
                        (Use src.graph.config.get_phase5_config() to get this)
        
        Returns:
            VisionConfig with all fields populated from YAML or defaults.
        
        Example:
            >>> from src.graph.config import load_entity_graph_config, get_phase5_config
            >>> cfg = load_entity_graph_config("configs/entity_graph.yaml")
            >>> phase5_cfg = get_phase5_config(cfg)
            >>> vision_cfg = VisionConfig.from_yaml_dict(phase5_cfg)
            >>> print(vision_cfg.model_id)
        """
        if not phase5_cfg:
            # Return default config if phase5 section missing (Phase 4 mode)
            return cls()
        
        # Extract nested sections with safe defaults
        detector = phase5_cfg.get("detector", {})
        candidates = phase5_cfg.get("candidates", {})
        safe_neighbors = phase5_cfg.get("safe_neighbors", {})
        negative_controls = phase5_cfg.get("negative_controls", {})
        thresholds = phase5_cfg.get("thresholds", {})
        binding = phase5_cfg.get("binding", {})
        pmi = phase5_cfg.get("pmi", {})
        
        # Flatten into VisionConfig fields
        return cls(
            # Detector
            model_id=detector.get("model_id", "owlvit"),
            batch_size_images=detector.get("batch_size_images", 8),
            batch_size_phrases=detector.get("batch_size_phrases", 128),
            cache_dir=detector.get("cache_dir", "data/vision/cache"),
            raw_shards_dir=detector.get("raw_shards_dir", "data/vision/raw_shards"),
            post_shards_dir=detector.get("post_shards_dir", "data/vision/post_shards"),
            shard_size=detector.get("shard_size", 500),
            
            # Candidates
            vis_prior_topK=candidates.get("vis_prior_topK", 50),
            tau_vis=candidates.get("tau_vis", 0.20),
            C_max=candidates.get("C_max", 128),
            
            # Safe neighbors
            tau_sim=safe_neighbors.get("tau_sim", 0.25),
            use_mutual_knn=safe_neighbors.get("use_mutual_knn", True),
            min_df_vis=safe_neighbors.get("min_df_vis", 5),
            require_in_vis_prior=safe_neighbors.get("require_in_vis_prior", False),
            
            # Negative controls
            null_phrases=negative_controls.get("null_phrases", [
                "xyzzy nonsense phrase 1",
                "blark random phrase 2"
            ]),
            noise_percentile=negative_controls.get("noise_percentile", 95.0),
            delta=negative_controls.get("delta", 0.05),
            
            # Thresholds
            threshold_strategy=thresholds.get("strategy", "grouped"),
            threshold_objective=thresholds.get("objective", "fixed_fpr"),
            target_fpr=thresholds.get("target_fpr", 0.05),
            target_precision=thresholds.get("target_precision", 0.90),
            min_support=thresholds.get("min_support", 200),
            
            # Binding
            binding_enable=binding.get("enable", False),
            binding_topN_images=binding.get("topN_images", 50),
            binding_topB_boxes=binding.get("topB_boxes", 5),
            hsv_ranges=binding.get("hsv_ranges", {}),
            clip_fallback_enable=binding.get("clip_fallback", {}).get("enable", True),
            clip_fallback_template=binding.get("clip_fallback", {}).get(
                "prompt_template", "a photo of a {attr} {obj}"
            ),
            
            # PMI
            pmi_smoothing_eps=pmi.get("smoothing_eps", 1.0),
            pmi_npmi_min=pmi.get("npmi_min", 0.20),
            pmi_npmi_min_stop=pmi.get("npmi_min_stop", 0.20),
            pmi_degree_cap=pmi.get("degree_cap", 500),
            pmi_per_node_topk=pmi.get("per_node_topk", 200),
            pmi_stop_nodes_topM_df=pmi.get("stop_nodes_topM_df", 50),
        )
