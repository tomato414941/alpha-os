"""Backward-compatible alias for `hypothesis_seeder`."""

from .hypothesis_seeder import HypothesisSeederDaemon, SeedingRoundStats

UnifiedAlphaGeneratorDaemon = HypothesisSeederDaemon
GenerationRoundStats = SeedingRoundStats
