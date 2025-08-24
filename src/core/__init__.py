"""
Core pipeline architecture for FinMapReduce.
"""

from .base_pipeline import BasePipeline
from .mapreduce_pipeline import MapReducePipeline
from .truncation_pipeline import TruncationPipeline
from .factory import PipelineFactory

__all__ = [
    'BasePipeline',
    'MapReducePipeline',
    'TruncationPipeline',
    'PipelineFactory'
]
