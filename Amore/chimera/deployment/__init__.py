"""
chimera.deployment — hermes-agent integration layer.

Provides:
  ChimeraContextEngine — hermes-agent ContextEngine plugin that trusts Mamba
    spans to handle long context and only fires summarization as a last resort.
  ThinkBridge — injects [THINK]/[/THINK] framing into hermes-agent prompts
    so the model runs a silent internal monologue before each visible response.

Usage in hermes-agent config.yaml:
  context:
    engine: chimera

Drop chimera/deployment/context_engine.py into the hermes-agent plugin path
(plugins/context_engine/chimera/__init__.py) and set the config key above.
"""

from .context_engine import ChimeraContextEngine
from .think_bridge import ThinkBridge

__all__ = ["ChimeraContextEngine", "ThinkBridge"]
