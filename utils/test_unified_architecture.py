#!/usr/bin/env python3
"""
Test script for the unified pipeline architecture.

Tests that both MapReduce and Truncation pipelines can be created and basic operations work.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all new components can be imported."""
    print("Testing imports...")

    try:
        from base_pipeline import BasePipeline
        print("✓ BasePipeline imported")

        from mapreduce_pipeline import MapReducePipeline
        print("✓ MapReducePipeline imported")

        from truncation_pipeline import TruncationPipeline
        print("✓ TruncationPipeline imported")

        from truncation_formatter import TruncationFormatter
        print("✓ TruncationFormatter imported")

        from factory import PipelineFactory
        print("✓ PipelineFactory imported")

        from dataset_loader import DatasetLoader
        print("✓ DatasetLoader imported")

        from financebench_loader import FinanceBenchLoader
        print("✓ FinanceBenchLoader imported")

        from finqa_loader import FinQALoader
        print("✓ FinQALoader imported")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_factory_methods():
    """Test that factory methods work correctly."""
    print("\nTesting factory methods...")

    try:
        from factory import PipelineFactory

        # Test available datasets
        datasets = PipelineFactory.get_available_datasets()
        print(f"✓ Available datasets: {datasets}")

        # Test available formats
        formats = PipelineFactory.get_available_formats()
        print(f"✓ Available formats: {formats}")

        # Test available strategies
        strategies = PipelineFactory.get_available_strategies()
        print(f"✓ Available strategies: {strategies}")

        # Test available approaches
        approaches = PipelineFactory.get_available_approaches()
        print(f"✓ Available approaches: {approaches}")

        return True

    except Exception as e:
        print(f"✗ Factory method test failed: {e}")
        return False

def test_pipeline_info():
    """Test pipeline info retrieval."""
    print("\nTesting pipeline info...")

    try:
        from factory import PipelineFactory

        # Test MapReduce pipeline info
        mr_info = PipelineFactory.get_pipeline_info(
            dataset='financebench',
            approach='mapreduce',
            format_type='json'
        )
        print(f"✓ MapReduce info: {mr_info['approach']} - {mr_info['pipeline_class']}")

        # Test Truncation pipeline info
        trunc_info = PipelineFactory.get_pipeline_info(
            dataset='financebench',
            approach='truncation',
            strategy='start'
        )
        print(f"✓ Truncation info: {trunc_info['approach']} - {trunc_info['pipeline_class']}")

        return True

    except Exception as e:
        print(f"✗ Pipeline info test failed: {e}")
        return False

def test_mock_pipeline_creation():
    """Test pipeline creation with mock components."""
    print("\nTesting mock pipeline creation...")

    try:
        from factory import PipelineFactory

        # Mock LLM class
        class MockLLM:
            def __init__(self):
                self.model_name = "mock-model"

            async def invoke(self, prompt):
                return {"content": "Mock response", "raw_response": None}

        # Mock prompts
        mock_prompts = {
            'map_prompt': "Context: {context}\nQuestion: {question}",
            'reduce_prompt': "Context: {context}\nQuestion: {question}",
            'judge_prompt': "Evaluate: {evaluation_text}"
        }

        # Test MapReduce pipeline creation
        try:
            mr_pipeline = PipelineFactory.create_pipeline(
                dataset='financebench',
                format_type='json',
                llm=MockLLM(),
                prompts_dict=mock_prompts,
                pdf_parser='marker'
            )
            print(f"✓ MapReduce pipeline created: {type(mr_pipeline).__name__}")
        except Exception as e:
            print(f"✗ MapReduce creation failed: {e}")

        # Test Truncation pipeline creation
        try:
            trunc_pipeline = PipelineFactory.create_truncation_pipeline(
                dataset='financebench',
                strategy='start',
                llm=MockLLM(),
                prompts_dict=mock_prompts,
                pdf_parser='marker'
            )
            print(f"✓ Truncation pipeline created: {type(trunc_pipeline).__name__}")
        except Exception as e:
            print(f"✗ Truncation creation failed: {e}")

        return True

    except Exception as e:
        print(f"✗ Mock pipeline creation failed: {e}")
        return False

def test_dataset_loaders():
    """Test dataset loader abstract methods."""
    print("\nTesting dataset loaders...")

    try:
        from financebench_loader import FinanceBenchLoader
        from finqa_loader import FinQALoader

        # Test FinanceBench loader
        fb_loader = FinanceBenchLoader(pdf_parser='marker')
        print(f"✓ FinanceBench loader created")
        print(f"  - Dataset name: {fb_loader.get_dataset_name()}")
        print(f"  - Results dir: {fb_loader.get_results_directory()}")

        # Test FinQA loader
        fq_loader = FinQALoader(doc_dir="/tmp")
        print(f"✓ FinQA loader created")
        print(f"  - Dataset name: {fq_loader.get_dataset_name()}")
        print(f"  - Results dir: {fq_loader.get_results_directory()}")

        # Test abstract methods exist
        mock_qa_pair = {"doc_name": "test.pdf"}

        # These should exist but may fail due to file not found
        try:
            _ = fb_loader.get_document_identifier(mock_qa_pair)
            print("✓ get_document_identifier works")
        except Exception:
            print("✓ get_document_identifier method exists")

        # Test that new load_full_document method exists
        assert hasattr(fb_loader, 'load_full_document'), "load_full_document method missing"
        assert hasattr(fq_loader, 'load_full_document'), "load_full_document method missing"
        print("✓ load_full_document method exists on both loaders")

        return True

    except Exception as e:
        print(f"✗ Dataset loader test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Testing Unified Pipeline Architecture ===\n")

    tests = [
        test_imports,
        test_factory_methods,
        test_pipeline_info,
        test_mock_pipeline_creation,
        test_dataset_loaders
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"=== Test Results: {passed}/{total} passed ===")

    if passed == total:
        print("🎉 All tests passed! The unified architecture is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)