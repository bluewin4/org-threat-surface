from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import argparse
import math
import os
import random

import networkx as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


VECTOR_DIM = 10
RNG = np.random.default_rng(seed=42)


class EdgeType(str, Enum):
    REPORTING = "REPORTING"
    EDICT = "EDICT"


class PsychProfile(str, Enum):
    NORMAL = "NORMAL"
    MISALIGNED = "MISALIGNED"


class ActionType(str, Enum):
    EMAIL = "email"
    EDICT = "edict"
    NONE = "none"


class SimulationConfig(BaseModel):
    num_agents: int = Field(default=30, ge=4)
    max_depth: int = Field(default=3, ge=1)
    branching_factor: int = Field(default=3, ge=1)
    friction: float = 0.5
    base_edict_cost: int = 20
    hop_cost: float = 1.0
    email_base_cost: float = 1.0
    edict_weight: float = 0.1
    global_safety_scalar: float = 0.5
    utility_threshold: float = 0.1
    token_replenish: int = 5
    max_tokens: int = 100
    mean_personal_weight: float = 0.3


@dataclass
class ActionLog:
    step: int
    agent_id: str
    action_type: ActionType
    source: str
    target: Optional[str]
    cost: float
    utility: float
    topology_change: bool
    s_org: float
    s_personal: float
    s_global: float
    shadow_link: bool
    w_org: float
    w_personal: float
    w_global: float


class OrganizationModel:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.graph: nx.DiGraph = nx.DiGraph()
        self.mission_vector: np.ndarray = self._random_unit_vector()
        self._path_cache: Dict[str, Dict[str, float]] = {}
        self._init_hierarchy()
        self._compute_centrality()

    # ----- Graph construction -------------------------------------------------
    @staticmethod
    def _random_unit_vector() -> np.ndarray:
        v = RNG.normal(size=VECTOR_DIM)
        norm = np.linalg.norm(v)
        if norm == 0:
            return np.zeros(VECTOR_DIM)
        return v / norm

    def _init_hierarchy(self) -> None:
        """
        Build a simple hierarchical org:
        level 0: CEO
        level 1..max_depth: managers/ICs
        """
        # Create CEO
        ceo_id = "CEO"
        self.graph.add_node(
            ceo_id,
            role="CEO",
            vector=self._random_unit_vector(),
            level=0,
        )

        current_level_nodes = [ceo_id]
        next_id = 0

        for level in range(1, self.config.max_depth + 1):
            next_level_nodes: List[str] = []
            for parent in current_level_nodes:
                for _ in range(self.config.branching_factor):
                    if len(self.graph.nodes) >= self.config.num_agents:
                        break
                    node_id = f"N{next_id}"
                    next_id += 1
                    role = self._role_for_level(level)
                    self.graph.add_node(
                        node_id,
                        role=role,
                        vector=self._random_unit_vector(),
                        level=level,
                    )
                    # Reporting edge: parent -> child (downward)
                    self.graph.add_edge(
                        parent,
                        node_id,
                        weight=1.0,
                        type=EdgeType.REPORTING.value,
                    )
                    # Optional upward communication edge with slightly higher friction
                    self.graph.add_edge(
                        node_id,
                        parent,
                        weight=1.5,
                        type=EdgeType.REPORTING.value,
                    )
                    next_level_nodes.append(node_id)
                if len(self.graph.nodes) >= self.config.num_agents:
                    break
            current_level_nodes = next_level_nodes
            if not current_level_nodes:
                break

    def _role_for_level(self, level: int) -> str:
        if level == 1:
            return "VP"
        if level == 2:
            return "Director"
        return "IC"

    def _compute_centrality(self) -> None:
        # Betweenness centrality as a proxy for influence
        centrality = nx.betweenness_centrality(self.graph, normalized=True)
        nx.set_node_attributes(self.graph, centrality, "centrality")

    # ----- Costs -------------------------------------------------------------
    def get_distance_cost(self, source: str, target: str) -> float:
        """
        Distance cost based on shortest path length * hop_cost,
        with direction-aware friction so messaging down is cheaper than up.
        Cached to speed up frequent queries; invalidated on topology changes.
        """
        # Check cache first
        if source in self._path_cache and target in self._path_cache[source]:
            return self._path_cache[source][target]

        try:
            path = nx.shortest_path(self.graph, source=source, target=target)
        except nx.NetworkXNoPath:
            return math.inf

        # Compute "angle" via level differences; downward hops are cheaper
        total_cost = 0.0
        for u, v in zip(path[:-1], path[1:]):
            edge_data = self.graph.get_edge_data(u, v)
            weight = edge_data.get("weight", 1.0) if edge_data else 1.0
            level_u = self.graph.nodes[u].get("level", 0)
            level_v = self.graph.nodes[v].get("level", 0)
            # If going down the hierarchy, slight discount; if up, extra cost
            if level_v > level_u:
                direction_factor = 0.8
            elif level_v < level_u:
                direction_factor = 1.3
            else:
                direction_factor = 1.0
            total_cost += weight * direction_factor

        final_cost = self.config.hop_cost * total_cost
        
        # Cache the result
        if source not in self._path_cache:
            self._path_cache[source] = {}
        self._path_cache[source][target] = final_cost
        
        return final_cost

    def get_edict_cost(self, source: str, target: str) -> float:
        """
        Edict cost = base_cost + 5 * sum(centrality of bypassed nodes).
        Bypassed nodes are internal nodes on the shortest existing path.
        """
        try:
            path = nx.shortest_path(self.graph, source=source, target=target)
        except nx.NetworkXNoPath:
            # If no path exists, edict is extremely expensive but still possible
            return self.config.base_edict_cost * 2

        internal_nodes = path[1:-1]
        centrality_sum = sum(
            float(self.graph.nodes[n].get("centrality", 0.0)) for n in internal_nodes
        )
        return self.config.base_edict_cost + 5.0 * centrality_sum

    def apply_edict(self, source: str, target: str) -> None:
        """
        Create a low-resistance direct edge (wormhole) between source and target.
        """
        self.graph.add_edge(
            source,
            target,
            weight=self.config.edict_weight,
            type=EdgeType.EDICT.value,
        )
        # Recompute centrality to reflect new topology
        self._compute_centrality()
        # Invalidate path cache since topology changed
        self._path_cache.clear()

    # ----- Utility helpers ---------------------------------------------------
    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            return 0.0
        return float(np.dot(v1, v2) / denom)

    def node_vector(self, node: str) -> np.ndarray:
        return self.graph.nodes[node]["vector"]

    def node_centrality(self, node: str) -> float:
        return float(self.graph.nodes[node].get("centrality", 0.0))

    def previous_distance(self, source: str, target: str) -> int:
        try:
            return nx.shortest_path_length(self.graph, source=source, target=target)
        except nx.NetworkXNoPath:
            return math.inf

    def org_gain_weight(self, node: str) -> float:
        """
        Per-agent weighting for organizational gain:
        - Normalized by org size (number of nodes)
        - Higher-level nodes (closer to top) receive greater weight.
        """
        n = self.graph.number_of_nodes()
        if n == 0:
            return 0.0
        levels = nx.get_node_attributes(self.graph, "level") or {}
        if not levels:
            return 1.0 / float(n)
        max_level = max(levels.values())
        node_level = levels.get(node, 0)
        # Higher in the hierarchy (smaller level) gets larger base weight.
        base = float(max_level - node_level + 1)
        # Normalize by org size so that individual org gain shrinks in larger orgs.
        return base / float(n)

    # ----- Visualization & detection ----------------------------------------
    def render_graph(
        self,
        path: Optional[str] = None,
        layout: str = "spring",
    ) -> None:
        import matplotlib.pyplot as plt

        if layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            # Spring layout with edge weights used as attractive forces
            pos = nx.spring_layout(self.graph, seed=42, weight="weight")

        reporting_edges = [
            (u, v)
            for u, v, d in self.graph.edges(data=True)
            if d.get("type") == EdgeType.REPORTING.value
        ]
        edict_edges = []
        for u, v, d in self.graph.edges(data=True):
            if d.get("type") == EdgeType.EDICT.value:
                edict_edges.append((u, v))

        # Shadow links: edict edges that bypass long prior paths
        suspicious_set = set(self.detect_suspicious_edicts())
        shadow_edges = [e for e in edict_edges if e in suspicious_set]
        non_shadow_edicts = [e for e in edict_edges if e not in suspicious_set]

        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color="lightblue",
            node_size=300,
        )
        nx.draw_networkx_labels(self.graph, pos, font_size=8)
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=reporting_edges,
            edge_color="gray",
            arrows=True,
            alpha=0.5,
        )
        # Normal edict edges (solid red)
        if non_shadow_edicts:
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=non_shadow_edicts,
                edge_color="red",
                arrows=True,
                width=2.0,
            )

        # Shadow links: red, dotted
        if shadow_edges:
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=shadow_edges,
                edge_color="red",
                style="dotted",
                arrows=True,
                width=2.5,
            )
        plt.axis("off")
        if path:
            plt.tight_layout()
            plt.savefig(path)
        else:
            plt.show()
        plt.close()

    def detect_suspicious_edicts(self) -> List[Tuple[str, str]]:
        """
        Flag any edict edge where the previous distance between nodes was > 3.
        We approximate 'previous' by temporarily removing the edict edge.
        """
        suspicious: List[Tuple[str, str]] = []
        for u, v, d in list(self.graph.edges(data=True)):
            if d.get("type") != EdgeType.EDICT.value:
                continue
            # Temporarily remove and measure distance
            self.graph.remove_edge(u, v)
            dist = self.previous_distance(u, v)
            if dist > 3:
                suspicious.append((u, v))
            # Restore edge
            self.graph.add_edge(u, v, **d)
        return suspicious


class Agent:
    def __init__(
        self,
        agent_id: str,
        role: str,
        node_id: str,
        psych_profile: PsychProfile,
        gain_weights: Dict[str, float],
        alignment_vector: np.ndarray,
        mission_vector: np.ndarray,
        secret_target: Optional[str] = None,
        token_budget: int = 100,
        learning_rate: float = 0.05,
    ):
        self.id = agent_id
        self.role = role
        self.node_id = node_id
        self.psych_profile = psych_profile
        self.gain_weights = gain_weights
        self.alignment_vector = alignment_vector
        self.mission_vector = mission_vector
        self.secret_target = secret_target
        self.tokens = token_budget
        self.learning_rate = learning_rate

    # ----- Utility computation ----------------------------------------------
    def _org_score(self, org: OrganizationModel, target: str) -> float:
        target_vec = org.node_vector(target)
        raw = org.cosine_similarity(target_vec, self.mission_vector)
        # Scale by org-wide gain weight for this agent's position in the hierarchy
        # so top-level roles derive more organizational gain, normalized by org size.
        weight = org.org_gain_weight(self.node_id)
        return raw * weight

    def _personal_score(self, org: OrganizationModel, target: str) -> float:
        if not self.secret_target:
            return 0.0
        try:
            dist = nx.shortest_path_length(
                org.graph, source=target, target=self.secret_target
            )
        except nx.NetworkXNoPath:
            return -1.0
        # Inverse distance; closer is better
        return 1.0 / (1.0 + float(dist))

    def _global_score(self, org: OrganizationModel) -> float:
        # Simple dot with a global safety scalar and own alignment
        base = org.config.global_safety_scalar
        # Encourage alignment with mission for global score
        return base * org.cosine_similarity(self.alignment_vector, org.mission_vector)

    def _compute_utility(self, org: OrganizationModel, target: str) -> float:
        s_org, s_pers, s_glob = self._score_components(org, target)
        return (
            self.gain_weights.get("org", 0.0) * s_org
            + self.gain_weights.get("personal", 0.0) * s_pers
            + self.gain_weights.get("global", 0.0) * s_glob
        )

    def _score_components(
        self, org: OrganizationModel, target: str
    ) -> Tuple[float, float, float]:
        s_org = self._org_score(org, target)
        s_pers = self._personal_score(org, target)
        s_glob = self._global_score(org)
        return s_org, s_pers, s_glob

    def _update_gains(
        self,
        reward: float,
        s_org: float,
        s_pers: float,
        s_glob: float,
    ) -> None:
        """
        Simple policy-gradient-style update on gain weights.

        w <- w + alpha * reward * S
        then clipped to [0, 1] and re-normalized to sum to 1.
        """
        alpha = self.learning_rate
        grad = np.array([s_org, s_pers, s_glob], dtype=float)
        w = np.array(
            [
                self.gain_weights.get("org", 0.0),
                self.gain_weights.get("personal", 0.0),
                self.gain_weights.get("global", 0.0),
            ],
            dtype=float,
        )
        # No update if gradient is all zeros
        if not np.any(grad):
            return

        w = w + alpha * reward * grad
        w = np.clip(w, 0.0, 1.0)
        s = float(w.sum())
        if s > 0:
            w = w / s
        self.gain_weights["org"] = float(w[0])
        self.gain_weights["personal"] = float(w[1])
        self.gain_weights["global"] = float(w[2])

    # ----- Decision logic ----------------------------------------------------
    def decide_action(
        self, org: OrganizationModel
    ) -> Tuple[
        ActionType,
        Optional[str],
        float,
        float,
        bool,
        float,
        float,
        float,
        bool,
        float,
        float,
        float,
    ]:
        """
        Decide between sending an email, issuing an edict, or doing nothing.
        Returns: (action_type, target_node, cost, utility, topology_change)
        """
        if self.tokens <= 0:
            return (
                ActionType.NONE,
                None,
                0.0,
                0.0,
                False,
                0.0,
                0.0,
                0.0,
                False,
                self.gain_weights.get("org", 0.0),
                self.gain_weights.get("personal", 0.0),
                self.gain_weights.get("global", 0.0),
            )

        # 1. Choose target: for simplicity, pick node that maximizes utility
        best_target = None
        best_utility = -math.inf
        for node in org.graph.nodes:
            if node == self.node_id:
                continue
            u = self._compute_utility(org, node)
            if u > best_utility:
                best_utility = u
                best_target = node

        if best_target is None:
            return (
                ActionType.NONE,
                None,
                0.0,
                0.0,
                False,
                0.0,
                0.0,
                0.0,
                False,
                self.gain_weights.get("org", 0.0),
                self.gain_weights.get("personal", 0.0),
                self.gain_weights.get("global", 0.0),
            )

        # 2. Compute costs
        distance_cost = org.get_distance_cost(self.node_id, best_target)
        if math.isinf(distance_cost):
            email_cost = math.inf
        else:
            email_cost = org.config.email_base_cost + org.config.friction * distance_cost

        edict_cost = org.get_edict_cost(self.node_id, best_target)

        # 3. Selection logic
        action = ActionType.NONE
        action_cost = 0.0
        topology_change = False
        shadow_link = False

        if email_cost <= self.tokens:
            # Prefer cheaper email when feasible
            action = ActionType.EMAIL
            action_cost = email_cost
        else:
            # Email too expensive, consider edict
            if (
                edict_cost <= self.tokens
                and best_utility > org.config.utility_threshold
            ):
                action = ActionType.EDICT
                action_cost = edict_cost
                topology_change = True
            else:
                action = ActionType.NONE
                action_cost = 0.0

        # Misaligned agents are more willing to issue edicts even if email is affordable
        if (
            self.psych_profile == PsychProfile.MISALIGNED
            and best_target is not None
            and edict_cost <= self.tokens
            and best_utility > org.config.utility_threshold
        ):
            # Compare "bang for buck": utility per token
            email_eff = (
                best_utility / email_cost if email_cost not in (0.0, math.inf) else 0.0
            )
            edict_eff = best_utility / edict_cost if edict_cost > 0 else 0.0
            # If edict has comparable efficiency and distance is large, choose edict
            prev_dist = org.previous_distance(self.node_id, best_target)
            if prev_dist > 3 and edict_eff >= 0.5 * email_eff:
                action = ActionType.EDICT
                action_cost = edict_cost
                topology_change = True

        # 4. Apply effects
        s_org = 0.0
        s_pers = 0.0
        s_glob = 0.0
        if best_target is not None and action != ActionType.NONE:
            s_org, s_pers, s_glob = self._score_components(org, best_target)

        if action == ActionType.EDICT and best_target is not None:
            # Determine whether this edict represents a "shadow" link
            prev_dist = org.previous_distance(self.node_id, best_target)
            shadow_link = prev_dist > 3
            org.apply_edict(self.node_id, best_target)

        # Reinforcement-style gain update based on realized reward
        if action != ActionType.NONE and best_target is not None:
            # Reward: intrinsic utility minus small cost penalty
            reward = best_utility - 0.01 * action_cost
            self._update_gains(reward, s_org, s_pers, s_glob)
        # Email is purely metadata in this toy model

        # Deduct tokens
        self.tokens -= int(math.ceil(action_cost))
        if self.tokens < 0:
            self.tokens = 0

        return (
            action,
            best_target,
            action_cost,
            best_utility,
            topology_change,
            s_org,
            s_pers,
            s_glob,
            shadow_link,
            self.gain_weights.get("org", 0.0),
            self.gain_weights.get("personal", 0.0),
            self.gain_weights.get("global", 0.0),
        )

    def replenish_tokens(self, config: SimulationConfig) -> None:
        self.tokens = min(self.tokens + config.token_replenish, config.max_tokens)


def _infer_level_from_snapshot_row(row: pd.Series) -> int:
    """
    Heuristic mapping from snapshot row to hierarchy level.
    Level 0 ~ CEO / board, then CxO, then business units, then others.
    """
    role_str = str(row.get("role", "")).lower()
    board = int(row.get("board", 0) or 0)
    ceoprez = int(row.get("ceoprez", 0) or 0)
    cxo = int(row.get("cxo", 0) or 0)
    bu = int(row.get("bu", 0) or 0)
    primary = int(row.get("primary", 0) or 0)
    support = int(row.get("support", 0) or 0)

    if ceoprez == 1 or "chief executive officer" in role_str or "ceo" in role_str:
        return 0
    if board == 1:
        return 0
    if cxo == 1:
        return 1
    if bu == 1:
        return 2
    if primary == 1:
        return 2
    if support == 1:
        return 3
    # Fallback for anything else
    return 3


def build_org_from_snapshot(
    config: SimulationConfig,
    snapshot_path: str = "snapshot.csv",
    company: Optional[str] = None,
    year: Optional[int] = None,
    max_nodes: Optional[int] = None,
) -> OrganizationModel:
    """
    Build an OrganizationModel from the Public Org-Structure snapshot.csv data.
    Approximates a hierarchy for a single (company, year) by inferring levels
    from titles and flags, then wiring edges from higher- to lower-level execs.
    """
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"snapshot file not found at {snapshot_path}")

    df = pd.read_csv(snapshot_path)

    if company is not None:
        df = df[df["company"].str.lower() == company.lower()]
    if year is not None:
        df = df[df["year"] == year]

    if df.empty:
        raise ValueError("No rows found for given filters in snapshot.csv")

    # If still multiple (company, year) combos, sample one group
    grouped = df.groupby(["company", "year"])
    if company is None or year is None:
        groups = list(grouped)
        chosen_key, df_group = random.choice(groups)
        df = df_group
    else:
        # Use the filtered subset as is
        df = grouped.get_group((company, year)) if (company, year) in grouped.groups else df

    if max_nodes is None:
        max_nodes = config.num_agents
    if len(df) > max_nodes:
        df = df.sample(n=max_nodes, random_state=42)
    df = df.reset_index(drop=True)

    G = nx.DiGraph()

    # Create nodes with inferred levels
    node_ids: List[str] = []
    for idx, row in df.iterrows():
        full_name = str(row.get("full_name", f"Exec {idx}"))
        role = str(row.get("role", "Executive"))
        node_id = f"{full_name} [{idx}]"
        level = _infer_level_from_snapshot_row(row)
        G.add_node(
            node_id,
            role=role,
            vector=OrganizationModel._random_unit_vector(),
            level=level,
        )
        node_ids.append(node_id)

    # Wire edges: for each non-root, connect to a higher-level node (if any)
    for idx, node_id in enumerate(node_ids):
        level = G.nodes[node_id].get("level", 0)
        if level == 0:
            continue
        # Potential parents: any previously created node with a lower level
        prior_nodes = node_ids[:idx]
        candidates = [n for n in prior_nodes if G.nodes[n].get("level", 0) < level]
        if not candidates:
            candidates = prior_nodes
        if not candidates:
            continue
        parent = random.choice(candidates)
        if parent == node_id:
            continue
        # Reporting edge: parent -> child (downward)
        G.add_edge(
            parent,
            node_id,
            weight=1.0,
            type=EdgeType.REPORTING.value,
        )
        # Upward communication edge
        G.add_edge(
            node_id,
            parent,
            weight=1.5,
            type=EdgeType.REPORTING.value,
        )

    # Build OrganizationModel from this pre-constructed graph
    mission_vec = OrganizationModel._random_unit_vector()
    org = OrganizationModel.__new__(OrganizationModel)  # bypass __init__
    org.config = config
    org.graph = G
    org.mission_vector = mission_vec
    org._path_cache = {}
    org._compute_centrality()
    return org


def build_org_from_master(
    config: SimulationConfig,
    master_path: str = "master_SP500_TMT.csv",
    company: Optional[str] = None,
    gv_key: Optional[int] = None,
    year: Optional[int] = None,
    max_nodes: Optional[int] = None,
) -> OrganizationModel:
    """
    Build an OrganizationModel from the full SP500 TMT master dataset.
    Uses the same level/role heuristics as snapshot-based builder but
    groups by (GV_KEY, year) to approximate a firm's top management team
    in a given fiscal year.
    """
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"master file not found at {master_path}")

    df = pd.read_csv(master_path)

    if gv_key is not None:
        df = df[df["GV_KEY"] == gv_key]
    if company is not None:
        df = df[df["company"].str.lower() == company.lower()]
    if year is not None:
        df = df[df["year"] == year]

    if df.empty:
        raise ValueError("No rows found for given filters in master_SP500_TMT.csv")

    if max_nodes is None:
        max_nodes = config.num_agents

    # Group by firm-year; if multiple, randomly choose one group
    grouped = df.groupby(["GV_KEY", "year"])
    groups = list(grouped)
    if len(groups) == 0:
        raise ValueError("No firm-year groups available after filtering master dataset")

    chosen_key, df_group = random.choice(groups)
    df = df_group

    if len(df) > max_nodes:
        df = df.sample(n=max_nodes, random_state=42)
    df = df.reset_index(drop=True)

    G = nx.DiGraph()

    node_ids: List[str] = []
    for idx, row in df.iterrows():
        full_name = str(row.get("full_name", f"Exec {idx}"))
        role = str(row.get("role", "Executive"))
        node_id = f"{full_name} [{idx}]"
        level = _infer_level_from_snapshot_row(row)
        G.add_node(
            node_id,
            role=role,
            vector=OrganizationModel._random_unit_vector(),
            level=level,
        )
        node_ids.append(node_id)

    # Wire approximate hierarchy within the TMT: higher levels to lower levels
    for idx, node_id in enumerate(node_ids):
        level = G.nodes[node_id].get("level", 0)
        if level == 0:
            continue
        prior_nodes = node_ids[:idx]
        candidates = [n for n in prior_nodes if G.nodes[n].get("level", 0) < level]
        if not candidates:
            candidates = prior_nodes
        if not candidates:
            continue
        parent = random.choice(candidates)
        if parent == node_id:
            continue
        G.add_edge(
            parent,
            node_id,
            weight=1.0,
            type=EdgeType.REPORTING.value,
        )
        G.add_edge(
            node_id,
            parent,
            weight=1.5,
            type=EdgeType.REPORTING.value,
        )

    mission_vec = OrganizationModel._random_unit_vector()
    org = OrganizationModel.__new__(OrganizationModel)
    org.config = config
    org.graph = G
    org.mission_vector = mission_vec
    org._path_cache = {}
    org._compute_centrality()
    return org


def compute_step_reward_series(
    logs: List[ActionLog], steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average organizational and personal rewards per step,
    using the current gain weights as multipliers on score components.
    """
    org_series = np.full(steps, np.nan, dtype=float)
    pers_series = np.full(steps, np.nan, dtype=float)

    by_step: Dict[int, List[ActionLog]] = {}
    for log in logs:
        if log.action_type == ActionType.NONE or log.target is None:
            continue
        by_step.setdefault(log.step, []).append(log)

    for step, entries in by_step.items():
        org_vals = [e.s_org * e.w_org for e in entries]
        pers_vals = [e.s_personal * e.w_personal for e in entries]
        if org_vals:
            org_series[step] = float(np.mean(org_vals))
            pers_series[step] = float(np.mean(pers_vals))

    return org_series, pers_series


def plot_reward_dynamics(
    org_series: np.ndarray,
    pers_series: np.ndarray,
    path: Optional[str] = None,
) -> None:
    """
    Plot how average organizational vs personal scores evolve over time.
    """
    import matplotlib.pyplot as plt

    steps = np.arange(len(org_series))
    plt.figure(figsize=(8, 4))
    plt.plot(steps, org_series, label="Org score (avg per step)", color="blue")
    plt.plot(steps, pers_series, label="Personal score (avg per step)", color="orange")
    plt.axhline(0.0, color="gray", linewidth=0.5)
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()


def run_master_batch_experiment(
    master_path: str = "master_SP500_TMT.csv",
    n_orgs: int = 100,
    repeats: int = 100,
    steps: int = 100,
    max_nodes: Optional[int] = None,
    mean_personal_weight: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Run repeated simulations over many firm-year org structures from the
    master SP500 TMT dataset and aggregate reward dynamics and shadow links.
    Uses multiprocessing to parallelize runs over available CPU cores.

    Returns:
        org_avg: average org score per step across all runs
        pers_avg: average personal score per step across all runs
        total_shadow_links: total number of shadow links created across runs
    """
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"master file not found at {master_path}")

    df = pd.read_csv(master_path)
    pairs = df.groupby(["GV_KEY", "year"]).size().reset_index()[["GV_KEY", "year"]]
    if pairs.empty:
        raise ValueError("No firm-year pairs found in master_SP500_TMT.csv")

    sample = pairs.sample(
        n=min(n_orgs, len(pairs)),
        random_state=42,
    )

    tasks = []
    for _, row in sample.iterrows():
        gv = int(row["GV_KEY"])
        yr = int(row["year"])
        for _ in range(repeats):
            tasks.append((master_path, gv, yr, steps, max_nodes, mean_personal_weight))

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    # Use all available cores
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(_run_single_simulation_task, tasks))

    # Aggregate results
    org_sum = np.zeros(steps, dtype=float)
    pers_sum = np.zeros(steps, dtype=float)
    counts = np.zeros(steps, dtype=float)
    total_shadow_links = 0

    for shadow_count, org_series, pers_series in results:
        total_shadow_links += shadow_count
        for i in range(steps):
            if not math.isnan(org_series[i]):
                org_sum[i] += org_series[i]
                pers_sum[i] += pers_series[i]
                counts[i] += 1.0

    org_avg = np.divide(org_sum, counts, out=np.full_like(org_sum, np.nan), where=counts > 0)
    pers_avg = np.divide(pers_sum, counts, out=np.full_like(pers_sum, np.nan), where=counts > 0)
    return org_avg, pers_avg, total_shadow_links


def _run_single_simulation_task(args):
    """Helper for parallel execution."""
    master_path, gv, yr, steps, max_nodes, mean_personal_weight = args
    try:
        org, agents, logs = run_simulation(
            steps=steps,
            data_source="master",
            master_path=master_path,
            gv_key=gv,
            year=yr,
            max_nodes=max_nodes,
            mean_personal_weight=mean_personal_weight,
        )
        # Debug: ensure we have logs
        valid_actions = sum(1 for l in logs if l.action_type != ActionType.NONE)
        if valid_actions == 0:
            # Optional: print if a run had zero actions, might indicate issues
            # print(f"DEBUG: Org {gv}-{yr} had 0 valid actions.")
            pass
            
        shadow_links = sum(1 for l in logs if l.shadow_link)
        org_series, pers_series = compute_step_reward_series(logs, steps)
        return shadow_links, org_series, pers_series
    except Exception as e:
        # Print exception to stdout so we can see it in the terminal
        print(f"Simulation failed for {gv}-{yr}: {e}")
        # Return safe defaults on failure
        return 0, np.full(steps, np.nan), np.full(steps, np.nan)


def create_agents(org: OrganizationModel, config: SimulationConfig) -> List[Agent]:
    agents: List[Agent] = []
    nodes = list(org.graph.nodes)

    # Ensure we have some misaligned agents deep in the hierarchy
    deep_nodes = [n for n in nodes if org.graph.nodes[n].get("level", 0) >= 2]
    if not deep_nodes:
        deep_nodes = nodes[1:]

    # Choose a structurally important internal node as the personal secret target.
    # This is meant to mirror subtle, high-leverage units you might see in real org
    # structure datasets such as the SP500/TMT snapshots and codebook entries
    # from the Public Org-Structure Database (see `codebook.csv` and `snapshot.csv`).
    centralities = {n: org.node_centrality(n) for n in nodes}
    sorted_nodes = sorted(
        [n for n in nodes if n in centralities],
        key=lambda n: centralities[n],
        reverse=True,
    )
    # Prefer deep, high-centrality nodes; fall back to any non-CEO node.
    candidate_targets = [
        n for n in sorted_nodes if org.graph.nodes[n].get("level", 0) >= 2
    ] or sorted_nodes[1:]
    secret_target: Optional[str] = candidate_targets[0] if candidate_targets else None

    # Normal agents
    for node in nodes:
        role = org.graph.nodes[node].get("role", "IC")
        
        # Sample personal weight from a clipped normal distribution centered on the config mean
        w_pers = float(np.clip(RNG.normal(loc=config.mean_personal_weight, scale=0.1), 0.0, 1.0))
        
        # Distribute the remaining weight mostly to org, some to global
        remaining = 1.0 - w_pers
        w_org = remaining * random.uniform(0.5, 1.0)
        w_glob = remaining - w_org
        
        gain_weights = {
            "org": w_org,
            "personal": w_pers,
            "global": w_glob,
        }
        agent = Agent(
            agent_id=f"A_{node}",
            role=role,
            node_id=node,
            psych_profile=PsychProfile.NORMAL,
            gain_weights=gain_weights,
            alignment_vector=org._random_unit_vector(),
            mission_vector=org.mission_vector,
        )
        agents.append(agent)

    # Misaligned agents: choose 1–2 deep nodes
    num_misaligned = min(2, max(1, len(deep_nodes) // 5))
    misaligned_nodes = random.sample(deep_nodes, k=num_misaligned)
    for node in misaligned_nodes:
        role = org.graph.nodes[node].get("role", "IC")
        gain_weights = {
            "org": random.uniform(0.0, 0.2),
            "personal": random.uniform(0.8, 1.0),
            "global": random.uniform(-0.2, 0.2),
        }
        agent = Agent(
            agent_id=f"M_{node}",
            role=role,
            node_id=node,
            psych_profile=PsychProfile.MISALIGNED,
            gain_weights=gain_weights,
            alignment_vector=org._random_unit_vector(),
            mission_vector=org.mission_vector,
            secret_target=secret_target,
        )
        agents.append(agent)

    return agents


def run_simulation(
    steps: int = 50,
    data_source: str = "snapshot",
    snapshot_path: str = "snapshot.csv",
    master_path: str = "master_SP500_TMT.csv",
    company: Optional[str] = None,
    gv_key: Optional[int] = None,
    year: Optional[int] = None,
    max_nodes: Optional[int] = None,
    mean_personal_weight: float = 0.3,
) -> Tuple[OrganizationModel, List[Agent], List[ActionLog]]:
    """
    Run a simulation for a single organization.

    - data_source="snapshot": build org from snapshot.csv (Public Org-Structure Database).
    - data_source="master": build org from master_SP500_TMT.csv (full SP500 TMT dataset).
    - data_source="synthetic": generate a synthetic hierarchy.
    """
    config = SimulationConfig(mean_personal_weight=mean_personal_weight)

    if data_source == "snapshot":
        org = build_org_from_snapshot(
            config=config,
            snapshot_path=snapshot_path,
            company=company,
            year=year,
            max_nodes=max_nodes,
        )
    elif data_source == "master":
        org = build_org_from_master(
            config=config,
            master_path=master_path,
            company=company,
            gv_key=gv_key,
            year=year,
            max_nodes=max_nodes,
        )
    elif data_source == "synthetic":
        org = OrganizationModel(config)
    else:
        raise ValueError(f"Unknown data_source: {data_source}")
    agents = create_agents(org, config)
    logs: List[ActionLog] = []

    for t in range(steps):
        random.shuffle(agents)
        for agent in agents:
            (
                action,
                target,
                cost,
                utility,
                topo_change,
                s_org,
                s_pers,
                s_glob,
                shadow_link,
                w_org,
                w_pers,
                w_glob,
            ) = agent.decide_action(org)
            logs.append(
                ActionLog(
                    step=t,
                    agent_id=agent.id,
                    action_type=action,
                    source=agent.node_id,
                    target=target,
                    cost=cost,
                    utility=utility,
                    topology_change=topo_change,
                    s_org=s_org,
                    s_personal=s_pers,
                    s_global=s_glob,
                    shadow_link=shadow_link,
                    w_org=w_org,
                    w_personal=w_pers,
                    w_global=w_glob,
                )
            )
            agent.replenish_tokens(config)

    return org, agents, logs


def main(argv: Optional[List[str]] = None) -> None:
    """
    CLI entrypoint.

    Examples:
      - Single-org master run with visualization:
          python simulation.py --mode single --data-source master --steps 100 --max-nodes 50

      - Batch experiment over many master firm-years:
          python simulation.py --mode batch --steps 100 --n-orgs 100 --repeats 100 --max-nodes 50
    """
    parser = argparse.ArgumentParser(description="Org threat-surface toy model simulation")
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "sweep"],
        default="single",
        help="Run a single simulation, a batch experiment, or a parameter sweep.",
    )
    parser.add_argument(
        "--data-source",
        choices=["master", "snapshot", "synthetic"],
        default="master",
        help="Organization data source for single mode.",
    )
    parser.add_argument("--steps", type=int, default=100, help="Number of steps per simulation.")
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=50,
        help="Maximum number of nodes per org (for data-backed modes).",
    )
    parser.add_argument(
        "--master-path",
        type=str,
        default="master_SP500_TMT.csv",
        help="Path to master_SP500_TMT.csv.",
    )
    parser.add_argument(
        "--snapshot-path",
        type=str,
        default="snapshot.csv",
        help="Path to snapshot.csv.",
    )
    parser.add_argument(
        "--n-orgs",
        type=int,
        default=100,
        help="Number of distinct firm-year orgs to sample (batch mode).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="Number of repeats per org (batch mode).",
    )
    parser.add_argument(
        "--mean-personal-weight",
        type=float,
        default=0.3,
        help="Mean personal gain weight for normal agents.",
    )
    args = parser.parse_args(argv)

    if args.mode == "single":
        # ... single run logic ...
        if args.data_source == "master":
            org, agents, logs = run_simulation(
                steps=args.steps,
                data_source="master",
                master_path=args.master_path,
                max_nodes=args.max_nodes,
                mean_personal_weight=args.mean_personal_weight,
            )
        elif args.data_source == "snapshot":
            org, agents, logs = run_simulation(
                steps=args.steps,
                data_source="snapshot",
                snapshot_path=args.snapshot_path,
                max_nodes=args.max_nodes,
                mean_personal_weight=args.mean_personal_weight,
            )
        else:
            org, agents, logs = run_simulation(
                steps=args.steps,
                data_source="synthetic",
                max_nodes=args.max_nodes,
                mean_personal_weight=args.mean_personal_weight,
            )

        edicts = [log for log in logs if log.action_type == ActionType.EDICT]
        print(f"Total actions: {len(logs)}")
        print(f"Total edicts: {len(edicts)}")
        misaligned_edicts = [e for e in edicts if e.agent_id.startswith("M_")]
        print(f"Misaligned edicts: {len(misaligned_edicts)}")

        suspicious = org.detect_suspicious_edicts()
        print("Suspicious edicts (bypassing long distances > 3):")
        for u, v in suspicious:
            print(f"  {u} -> {v}")

        try:
            org.render_graph(layout="kamada_kawai")
        except Exception as exc:  # pragma: no cover - visualization issues
            print(f"Graph rendering failed: {exc}")

        org_series, pers_series = compute_step_reward_series(logs, args.steps)
        try:
            plot_reward_dynamics(org_series, pers_series)
        except Exception as exc:  # pragma: no cover - visualization issues
            print(f"Reward plotting failed: {exc}")

    elif args.mode == "batch":
        # Batch experiment over many master firm-years
        org_avg, pers_avg, total_shadow = run_master_batch_experiment(
            master_path=args.master_path,
            n_orgs=args.n_orgs,
            repeats=args.repeats,
            steps=args.steps,
            max_nodes=args.max_nodes,
            mean_personal_weight=args.mean_personal_weight,
        )
        print(f"Batch experiment complete over {args.n_orgs} orgs x {args.repeats} repeats.")
        print(f"Mean personal weight: {args.mean_personal_weight}")
        print(f"Total shadow links created: {total_shadow}")
        try:
            plot_reward_dynamics(org_avg, pers_avg)
        except Exception as exc:  # pragma: no cover - visualization issues
            print(f"Reward plotting failed: {exc}")

    elif args.mode == "sweep":
        # Parameter sweep mode
        weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results_org = []
        results_pers = []
        results_shadow = []

        print(f"Starting sweep over weights: {weights}")
        for w in weights:
            print(f"  Running batch for weight={w}...")
            org_avg, pers_avg, total_shadow = run_master_batch_experiment(
                master_path=args.master_path,
                n_orgs=args.n_orgs,
                repeats=args.repeats,
                steps=args.steps,
                max_nodes=args.max_nodes,
                mean_personal_weight=w,
            )
            # Take the mean of the last 10 steps to represent "final" stable state
            final_org = np.nanmean(org_avg[-10:]) if len(org_avg) >= 10 else np.nanmean(org_avg)
            final_pers = np.nanmean(pers_avg[-10:]) if len(pers_avg) >= 10 else np.nanmean(pers_avg)
            
            results_org.append(final_org)
            results_pers.append(final_pers)
            results_shadow.append(total_shadow)

        # Plot sweep results
        import matplotlib.pyplot as plt

        fig, ax1 = plt.figure(figsize=(10, 6)), plt.gca()

        # Plot rewards on left y-axis
        ax1.set_xlabel("Mean Personal Weight")
        ax1.set_ylabel("Final Average Reward", color="tab:blue")
        ax1.plot(weights, results_org, marker="o", label="Org Reward", color="tab:blue")
        ax1.plot(weights, results_pers, marker="s", label="Personal Reward", color="tab:cyan")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True, alpha=0.3)

        # Plot shadow links on right y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel("Total Shadow Links", color="tab:red")
        ax2.plot(weights, results_shadow, marker="^", linestyle="--", label="Shadow Links", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")

        plt.title("Impact of Personal Gain Weight on Rewards and Shadow Structure")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()


