"""
LLM narrative enrichment pipeline with structured output, rate limiting, and high-diversity fallback.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel

from smriti.generators.taxonomy import SubDomainMeta
from smriti.models.project_schema import SemanticContext, Taxonomy


class LLMEnricher:
    """
    Enriches project nodes with realistic technical narratives (problem, solution, tech stack, target audience).
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        ollama_endpoint: Optional[str] = None,
        prefer_offline: bool = False,
    ):
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.ollama_endpoint = ollama_endpoint or os.environ.get("OLLAMA_ENDPOINT")
        self.prefer_offline = prefer_offline

        # Action verbs, modifiers, and architectural motifs for high-diversity synthesis
        self._action_verbs = [
            "Distributed", "Accelerated", "Zero-Copy", "High-Throughput", "Fault-Tolerant",
            "Autonomous", "Predictive", "Adaptive", "Decentralized", "Hermetic",
            "Multi-Tenant", "Low-Latency", "Continuous", "Declarative", "Event-Driven",
        ]
        self._metric_units = [
            ("latency", ["p99 latency", "p95 tail latency", "round-trip time", "execution lag"]),
            ("throughput", ["events/sec", "queries/sec", "MB/s network throughput", "concurrent connections"]),
            ("scale", ["TB memory footprint", "multi-region nodes", "microservices", "active contributors"]),
        ]

    def enrich(
        self,
        taxonomy: Taxonomy,
        sub_domain_meta: SubDomainMeta,
        rice_reach_val: float,
        index: int,
    ) -> SemanticContext:
        """
        Enriches a project node with detailed narrative context.
        Uses live LLM if configured, otherwise falls back to deterministic synthesizer.
        """
        if not self.prefer_offline and self.gemini_api_key:
            try:
                return self._enrich_gemini(taxonomy, sub_domain_meta)
            except Exception as e:
                # Fallback on failure
                pass

        if not self.prefer_offline and self.ollama_endpoint:
            try:
                return self._enrich_ollama(taxonomy, sub_domain_meta)
            except Exception as e:
                # Fallback on failure
                pass

        return self._enrich_deterministic(taxonomy, sub_domain_meta, rice_reach_val, index)

    def _enrich_gemini(self, taxonomy: Taxonomy, sub_domain_meta: SubDomainMeta) -> SemanticContext:
        """
        Calls Gemini API with structured JSON output schema.
        """
        prompt = (
            f"Generate a realistic enterprise engineering project specification in JSON.\n"
            f"Domain: {taxonomy.domain}\n"
            f"Sub-domain: {taxonomy.sub_domain}\n"
            f"Epic: {taxonomy.epic}\n"
            f"Return JSON strictly with fields: title (str), problem_statement (str), solution_proposal (str), tech_stack (list of str), target_users (str)."
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.7,
            },
        }
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(raw_text)
            return SemanticContext(
                title=parsed["title"],
                problem_statement=parsed["problem_statement"],
                solution_proposal=parsed["solution_proposal"],
                tech_stack=parsed.get("tech_stack", sub_domain_meta.tech_stacks[0]),
                target_users=parsed.get("target_users", sub_domain_meta.target_audiences[0]),
            )

    def _enrich_ollama(self, taxonomy: Taxonomy, sub_domain_meta: SubDomainMeta) -> SemanticContext:
        """
        Calls local Ollama instance with structured prompt.
        """
        prompt = (
            f"You are a Principal Software Architect. Generate a concise enterprise project specification in JSON format.\n"
            f"Domain: {taxonomy.domain}, Sub-domain: {taxonomy.sub_domain}, Epic: {taxonomy.epic}.\n"
            f"Output JSON with keys: 'title', 'problem_statement', 'solution_proposal', 'tech_stack' (array of strings), 'target_users'."
        )
        url = f"{self.ollama_endpoint.rstrip('/')}/api/generate"
        payload = {
            "model": "llama3.2",
            "prompt": prompt,
            "format": "json",
            "stream": False,
        }
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            parsed = resp.json().get("response", {})
            if isinstance(parsed, str):
                import json
                parsed = json.loads(parsed)
            return SemanticContext(
                title=parsed["title"],
                problem_statement=parsed["problem_statement"],
                solution_proposal=parsed["solution_proposal"],
                tech_stack=parsed.get("tech_stack", sub_domain_meta.tech_stacks[0]),
                target_users=parsed.get("target_users", sub_domain_meta.target_audiences[0]),
            )

    def _enrich_deterministic(
        self,
        taxonomy: Taxonomy,
        sub_domain_meta: SubDomainMeta,
        reach_val: float,
        index: int,
    ) -> SemanticContext:
        """
        High-diversity deterministic narrative generator producing distinct, realistic technical specs.
        """
        tech_stack = random.choice(sub_domain_meta.tech_stacks)
        target_users = random.choice(sub_domain_meta.target_audiences)

        problem_template = random.choice(sub_domain_meta.problem_archetypes)
        solution_template = random.choice(sub_domain_meta.solution_archetypes)

        # Diverse parameter substitutions
        times = [15, 30, 45, 60, 90, 120]
        percents = [25, 40, 50, 65, 80, 95]
        latencies = ["120ms", "450ms", "1.2s", "2.8s", "5.4s"]
        throughputs = ["50k", "150k", "500k", "1.2M", "3.5M"]

        t_val = str(random.choice(times))
        p_val = str(random.choice(percents))
        lat_val = str(random.choice(latencies))
        tp_val = str(random.choice(throughputs))

        # Format problem and solution strings
        problem_str = (
            problem_template
            .replace("{time}", t_val)
            .replace("{percent}", p_val)
            .replace("{latency}", lat_val)
            .replace("{throughput}", tp_val)
        )

        modifier = random.choice(self._action_verbs)
        primary_tech = tech_stack[0]
        secondary_tech = tech_stack[1] if len(tech_stack) > 1 else "Cloud"

        # Unique titles combining epic, action, and core technologies
        title_patterns = [
            f"{modifier} {taxonomy.epic} with {primary_tech} and {secondary_tech}",
            f"{taxonomy.epic} Platform for {sub_domain_meta.name}",
            f"{modifier} {sub_domain_meta.name} Pipeline powered by {primary_tech}",
            f"Next-Gen {taxonomy.epic} Infrastructure ({primary_tech}/{secondary_tech})",
            f"{modifier} Architecture for {taxonomy.epic}",
        ]
        title = title_patterns[index % len(title_patterns)]

        solution_str = f"{solution_template} Implemented using {', '.join(tech_stack[:3])} to support high reliability and scale."

        return SemanticContext(
            title=title,
            problem_statement=problem_str,
            solution_proposal=solution_str,
            tech_stack=tech_stack,
            target_users=target_users,
        )
