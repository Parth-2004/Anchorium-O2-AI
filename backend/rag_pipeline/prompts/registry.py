"""Agent Prompt Registry — central access point for all agent system prompts.

Ensures Core Directives are ALWAYS prepended to every agent's prompt.
This is the only sanctioned way to get a system prompt for any agent —
direct access to the raw constants in ``agents.py`` is discouraged
outside of testing.
"""

from __future__ import annotations

import structlog

from rag_pipeline.prompts.agents import AGENT_PROMPTS
from rag_pipeline.prompts.core_directives import CORE_DIRECTIVES

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Sentinel string that MUST appear in every assembled prompt.
_CORE_SENTINEL = "NON-NEGOTIABLE RULES:"


class AgentPromptRegistry:
    """Registry that assembles complete system prompts for Omni-Engine agents.

    Usage::

        registry = AgentPromptRegistry()
        prompt = registry.get_system_prompt("compliance_copilot")
        # → Core Directives + Compliance Copilot specific block
    """

    _valid_agents: frozenset[str] = frozenset(AGENT_PROMPTS.keys())

    def get_system_prompt(self, agent_name: str) -> str:
        """Return the full system prompt for the named agent.

        The returned string is always:
            Core Directives + "\\n\\n" + agent-specific block

        Args:
            agent_name: One of the registered agent names.

        Returns:
            Complete system prompt string.

        Raises:
            ValueError: If ``agent_name`` is not registered.
        """
        if agent_name not in self._valid_agents:
            raise ValueError(f"Unknown agent '{agent_name}'. Valid agents: {sorted(self._valid_agents)}")

        agent_prompt = AGENT_PROMPTS[agent_name]
        assembled = f"{CORE_DIRECTIVES}\n\n{agent_prompt}"

        # Defensive: verify Core Directives are actually present
        if _CORE_SENTINEL not in assembled:
            logger.critical(
                "core_directives_missing_from_assembled_prompt",
                agent_name=agent_name,
            )
            raise RuntimeError(
                f"CRITICAL: Core Directives sentinel not found in assembled "
                f"prompt for agent '{agent_name}'. This is a configuration "
                f"error that must be fixed before any agent runs."
            )

        logger.info(
            "prompt_assembled",
            agent_name=agent_name,
            prompt_length=len(assembled),
        )
        return assembled

    @classmethod
    def list_agents(cls) -> list[str]:
        """Return sorted list of all registered agent names."""
        return sorted(cls._valid_agents)

    @classmethod
    def validate_all(cls) -> dict[str, bool]:
        """Validate that all registered agents produce valid prompts.

        Returns:
            Dict mapping agent name → True if valid, False if broken.
        """
        registry = cls()
        results: dict[str, bool] = {}
        for name in cls._valid_agents:
            try:
                prompt = registry.get_system_prompt(name)
                results[name] = _CORE_SENTINEL in prompt
            except Exception:
                results[name] = False
        return results
