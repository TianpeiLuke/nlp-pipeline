"""
Pytest tests for PydanticAI RnR Reason Code classification.
Tests to verify the implementation works correctly.
"""

import pytest
import asyncio
import logging
from typing import Dict, Any
from unittest.mock import Mock, patch

# Import the modules to test
from src.pydantic_ai.bedrock_rnr_agent import BedrockRnRPydanticAgent
from src.pydantic_ai.rnr_bedrock_main import RnRPydanticBedrockProcessor
from src.pydantic_ai.rnr_reason_code_models import (
    ReasonCodeCategory, 
    RnRReasonCodeAnalysis, 
    EvidenceSection, 
    ReasoningSection,
    RnRAnalysisInput,
    RnRCaseData
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPydanticModels:
    """Test Pydantic model validation and functionality."""
    
    def test_reason_code_category_enum(self):
        """Test ReasonCodeCategory enum values."""
        assert ReasonCodeCategory.TRUE_DNR == "TrueDNR"
        assert ReasonCodeCategory.CONFIRMED_DELAY == "Confirmed_Delay"
        assert ReasonCodeCategory.BUYER_CANCELLATION == "BuyerCancellation"
        assert ReasonCodeCategory.INSUFFICIENT_INFORMATION == "Insufficient_Information"
        
        # Test all categories are present
        expected_categories = [
            "TrueDNR", "Confirmed_Delay", "Delivery_Attempt_Failed", 
            "Seller_Unable_To_Ship", "PDA_Undeliverable", "PDA_Early_Refund",
            "Buyer_Received_WrongORDefective_Item", "Returnless_Refund", 
            "BuyerCancellation", "Return_NoLongerNeeded", 
            "Product_Information_Support", "Insufficient_Information"
        ]
        
        category_values = [cat.value for cat in ReasonCodeCategory]
        assert len(category_values) == 12
        for expected in expected_categories:
            assert expected in category_values
    
    def test_evidence_section_validation(self):
        """Test EvidenceSection model validation."""
        # Test with proper [sep] prefix
        evidence = EvidenceSection(
            message_evidence=["[sep] Test message"],
            shipping_evidence=["[sep] Test shipping"],
            timeline_evidence=["[sep] Test timeline"]
        )
        
        assert evidence.message_evidence == ["[sep] Test message"]
        assert evidence.shipping_evidence == ["[sep] Test shipping"]
        assert evidence.timeline_evidence == ["[sep] Test timeline"]
        
        # Test auto-formatting without [sep] prefix
        evidence_auto = EvidenceSection(
            message_evidence=["Test message without prefix"],
            shipping_evidence=["Test shipping without prefix"],
            timeline_evidence=["Test timeline without prefix"]
        )
        
        assert evidence_auto.message_evidence == ["[sep] Test message without prefix"]
        assert evidence_auto.shipping_evidence == ["[sep] Test shipping without prefix"]
        assert evidence_auto.timeline_evidence == ["[sep] Test timeline without prefix"]
    
    def test_reasoning_section_validation(self):
        """Test ReasoningSection model validation."""
        reasoning = ReasoningSection(
            primary_factors=["[sep] Primary factor"],
            supporting_evidence=["[sep] Supporting evidence"],
            contradicting_evidence=["[sep] None"]
        )
        
        assert reasoning.primary_factors == ["[sep] Primary factor"]
        assert reasoning.supporting_evidence == ["[sep] Supporting evidence"]
        assert reasoning.contradicting_evidence == ["[sep] None"]
    
    def test_rnr_analysis_model(self):
        """Test RnRReasonCodeAnalysis model."""
        evidence = EvidenceSection(
            message_evidence=["[sep] Test message"],
            shipping_evidence=["[sep] Test shipping"],
            timeline_evidence=["[sep] Test timeline"]
        )
        
        reasoning = ReasoningSection(
            primary_factors=["[sep] Primary factor"],
            supporting_evidence=["[sep] Supporting evidence"],
            contradicting_evidence=["[sep] None"]
        )
        
        analysis = RnRReasonCodeAnalysis(
            category=ReasonCodeCategory.BUYER_CANCELLATION,
            confidence_score=0.85,
            key_evidence=evidence,
            reasoning=reasoning
        )
        
        assert analysis.category == ReasonCodeCategory.BUYER_CANCELLATION
        assert analysis.confidence_score == 0.85
        assert isinstance(analysis.key_evidence, EvidenceSection)
        assert isinstance(analysis.reasoning, ReasoningSection)
    
    def test_confidence_score_validation(self):
        """Test confidence score validation and rounding."""
        evidence = EvidenceSection()
        reasoning = ReasoningSection()
        
        # Test valid range
        analysis = RnRReasonCodeAnalysis(
            category=ReasonCodeCategory.BUYER_CANCELLATION,
            confidence_score=0.856789,
            key_evidence=evidence,
            reasoning=reasoning
        )
        
        assert analysis.confidence_score == 0.86  # Should be rounded to 2 decimal places
        
        # Test boundary values
        analysis_min = RnRReasonCodeAnalysis(
            category=ReasonCodeCategory.BUYER_CANCELLATION,
            confidence_score=0.0,
            key_evidence=evidence,
            reasoning=reasoning
        )
        assert analysis_min.confidence_score == 0.0
        
        analysis_max = RnRReasonCodeAnalysis(
            category=ReasonCodeCategory.BUYER_CANCELLATION,
            confidence_score=1.0,
            key_evidence=evidence,
            reasoning=reasoning
        )
        assert analysis_max.confidence_score == 1.0
        
        # Test invalid values should raise validation error
        with pytest.raises(ValueError):
            RnRReasonCodeAnalysis(
                category=ReasonCodeCategory.BUYER_CANCELLATION,
                confidence_score=1.5,  # Invalid: > 1.0
                key_evidence=evidence,
                reasoning=reasoning
            )
        
        with pytest.raises(ValueError):
            RnRReasonCodeAnalysis(
                category=ReasonCodeCategory.BUYER_CANCELLATION,
                confidence_score=-0.1,  # Invalid: < 0.0
                key_evidence=evidence,
                reasoning=reasoning
            )
    
    def test_formatted_output(self):
        """Test formatted output generation."""
        evidence = EvidenceSection(
            message_evidence=["[sep] Buyer requests cancellation"],
            shipping_evidence=["[sep] No shipping events recorded"],
            timeline_evidence=["[sep] Request made before shipment"]
        )
        
        reasoning = ReasoningSection(
            primary_factors=["[sep] Buyer initiated cancellation before delivery"],
            supporting_evidence=["[sep] No shipping tracking available"],
            contradicting_evidence=["[sep] None"]
        )
        
        analysis = RnRReasonCodeAnalysis(
            category=ReasonCodeCategory.BUYER_CANCELLATION,
            confidence_score=0.92,
            key_evidence=evidence,
            reasoning=reasoning
        )
        
        formatted = analysis.to_formatted_output()
        
        # Check that formatted output contains expected sections
        assert "1. Category: BuyerCancellation" in formatted
        assert "2. Confidence Score: 0.92" in formatted
        assert "3. Key Evidence:" in formatted
        assert "4. Reasoning:" in formatted
        assert "* Message Evidence:" in formatted
        assert "* Shipping Evidence:" in formatted
        assert "* Timeline Evidence:" in formatted
        assert "* Primary Factors:" in formatted
        assert "* Supporting Evidence:" in formatted
        assert "* Contradicting Evidence:" in formatted
        
        # Check evidence content
        assert "[sep] Buyer requests cancellation" in formatted
        assert "[sep] No shipping events recorded" in formatted
        assert "[sep] Request made before shipment" in formatted
        
        # Check reasoning content
        assert "[sep] Buyer initiated cancellation before delivery" in formatted
        assert "[sep] No shipping tracking available" in formatted
        assert "[sep] None" in formatted
    
    def test_input_models(self):
        """Test input model validation."""
        # Test RnRAnalysisInput
        analysis_input = RnRAnalysisInput(
            dialogue="Test dialogue",
            shiptrack="Test shiptrack",
            max_estimated_arrival_date="2025-02-22"
        )
        
        assert analysis_input.dialogue == "Test dialogue"
        assert analysis_input.shiptrack == "Test shiptrack"
        assert analysis_input.max_estimated_arrival_date == "2025-02-22"
        
        # Test RnRCaseData
        case_data = RnRCaseData(
            dialogue="Test dialogue",
            shiptrack="Test shiptrack",
            max_estimated_arrival_date="2025-02-22",
            case_id="test_case_001",
            metadata={"source": "test"}
        )
        
        assert case_data.dialogue == "Test dialogue"
        assert case_data.shiptrack == "Test shiptrack"
        assert case_data.max_estimated_arrival_date == "2025-02-22"
        assert case_data.case_id == "test_case_001"
        assert case_data.metadata == {"source": "test"}


class TestBedrockRnRPydanticAgent:
    """Test BedrockRnRPydanticAgent functionality."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        with patch('src.pydantic_ai.bedrock_rnr_agent.boto3.client'):
            with patch('src.pydantic_ai.bedrock_rnr_agent.BedrockModel'):
                with patch('src.pydantic_ai.bedrock_rnr_agent.Agent'):
                    agent = BedrockRnRPydanticAgent(
                        model_id="test-model",
                        region_name="us-west-2",
                        use_inference_profile=False
                    )
                    return agent
    
    def test_agent_initialization(self, mock_agent):
        """Test agent initialization."""
        assert mock_agent.model_id == "test-model"
        assert mock_agent.region_name == "us-west-2"
        assert mock_agent.use_inference_profile == False
    
    def test_system_prompt_generation(self, mock_agent):
        """Test system prompt generation."""
        prompt = mock_agent._get_system_prompt()
        
        # Check that prompt contains key elements
        assert "You are an expert in analyzing buyer-seller messaging conversations" in prompt
        assert "TrueDNR (Delivered Not Received)" in prompt
        assert "Confirmed_Delay" in prompt
        assert "BuyerCancellation" in prompt
        assert "Insufficient_Information" in prompt
        assert "CRITICAL OUTPUT REQUIREMENTS" in prompt
        assert "CONFIDENCE SCORING GUIDELINES" in prompt
    
    def test_fallback_analysis_creation(self, mock_agent):
        """Test fallback analysis creation."""
        error_msg = "Test error message"
        fallback = mock_agent._create_fallback_analysis(error_msg)
        
        assert isinstance(fallback, RnRReasonCodeAnalysis)
        assert fallback.category == ReasonCodeCategory.INSUFFICIENT_INFORMATION
        assert fallback.confidence_score == 0.0
        assert f"[sep] Error processing case: {error_msg}" in fallback.key_evidence.message_evidence
        assert "[sep] Unable to parse shipping evidence due to error" in fallback.key_evidence.shipping_evidence
        assert "[sep] Unable to parse timeline evidence due to error" in fallback.key_evidence.timeline_evidence
        assert "[sep] Processing error prevented analysis" in fallback.reasoning.primary_factors
        assert f"[sep] Technical error: {error_msg}" in fallback.reasoning.supporting_evidence
        assert "[sep] None" in fallback.reasoning.contradicting_evidence
    
    def test_decision_tree_info(self, mock_agent):
        """Test decision tree info retrieval."""
        info = mock_agent.get_decision_tree_info()
        
        assert info["version"] == "PydanticAI-2.0"
        assert info["framework"] == "PydanticAI with AWS Bedrock"
        assert info["approach"] == "Feature-based Category Priority Hierarchy with Structured Output"
        
        # Check decision tree structure
        assert "decision_tree" in info
        assert "step_1" in info["decision_tree"]
        assert "step_2" in info["decision_tree"]
        assert "groups" in info["decision_tree"]
        
        # Check groups
        groups = info["decision_tree"]["groups"]
        assert "A" in groups
        assert "B" in groups
        assert "C" in groups
        
        # Check key criteria
        assert "key_criteria" in info
        assert "timeline_evidence" in info["key_criteria"]
        assert "return_record" in info["key_criteria"]
        assert "delivery_confirmation" in info["key_criteria"]
        
        # Check PydanticAI features
        assert "pydantic_ai_features" in info
        assert "structured_output" in info["pydantic_ai_features"]
        assert "type_safety" in info["pydantic_ai_features"]
        assert "async_support" in info["pydantic_ai_features"]
        assert "error_handling" in info["pydantic_ai_features"]


class TestRnRPydanticBedrockProcessor:
    """Test RnRPydanticBedrockProcessor functionality."""
    
    @pytest.fixture
    def mock_processor(self):
        """Create a mock processor for testing."""
        with patch('src.pydantic_ai.rnr_bedrock_main.BedrockRnRPydanticAgent'):
            processor = RnRPydanticBedrockProcessor(
                model_id="test-model",
                region_name="us-west-2",
                output_dir="./test_output",
                use_inference_profile=False
            )
            return processor
    
    def test_processor_initialization(self, mock_processor):
        """Test processor initialization."""
        assert mock_processor.model_id == "test-model"
        assert mock_processor.region_name == "us-west-2"
        assert mock_processor.use_inference_profile == False
        assert str(mock_processor.output_dir).endswith("test_output")
    
    def test_model_info(self, mock_processor):
        """Test model info retrieval."""
        info = mock_processor.get_model_info()
        
        assert info["original_model_id"] == "test-model"
        assert info["effective_model_id"] == "test-model"
        assert info["region_name"] == "us-west-2"
        assert info["use_inference_profile"] == False
        assert info["framework"] == "PydanticAI"
    
    def test_inference_profile_configs(self, mock_processor):
        """Test inference profile configurations."""
        # Test getting valid config
        config = mock_processor.get_inference_profile_config('claude-4-global')
        assert config["profile_id"] == 'global.anthropic.claude-sonnet-4-20250514-v1:0'
        assert config["model_id"] == 'anthropic.claude-sonnet-4-20250514-v1:0'
        assert config["region"] == 'us-east-1'
        
        # Test getting invalid config
        with pytest.raises(ValueError):
            mock_processor.get_inference_profile_config('invalid-config')
    
    @patch('pandas.read_csv')
    def test_load_data_from_file_csv(self, mock_read_csv, mock_processor):
        """Test loading data from CSV file."""
        # Mock pandas read_csv
        mock_df = Mock()
        mock_read_csv.return_value = mock_df
        
        # Mock Path.exists
        with patch('pathlib.Path.exists', return_value=True):
            result = mock_processor.load_data_from_file('test.csv')
            
            assert result == mock_df
            mock_read_csv.assert_called_once()
    
    def test_load_data_from_file_unsupported(self, mock_processor):
        """Test loading data from unsupported file format."""
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(ValueError, match="Unsupported file format"):
                mock_processor.load_data_from_file('test.xyz')
    
    def test_load_data_from_file_not_found(self, mock_processor):
        """Test loading data from non-existent file."""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="Input file not found"):
                mock_processor.load_data_from_file('nonexistent.csv')
    
    def test_validate_input_data(self, mock_processor):
        """Test input data validation."""
        import pandas as pd
        
        # Test valid data
        df = pd.DataFrame({
            'dialogue': ['Test dialogue 1', 'Test dialogue 2'],
            'shiptrack': ['Test shiptrack 1', 'Test shiptrack 2'],
            'max_estimated_arrival_date': ['2025-02-22', '2025-02-23']
        })
        
        validated_df = mock_processor.validate_input_data(df)
        
        assert 'dialogue' in validated_df.columns
        assert 'shiptrack' in validated_df.columns
        assert 'max_estimated_arrival_date' in validated_df.columns
        assert len(validated_df) == 2
        
        # Test missing required columns
        df_missing = pd.DataFrame({
            'dialogue': ['Test dialogue 1'],
            # Missing 'shiptrack' column
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            mock_processor.validate_input_data(df_missing)
    
    def test_process_single_case_success(self, mock_processor):
        """Test successful single case processing."""
        # Mock the agent's analyze_rnr_case_sync method
        mock_result = RnRReasonCodeAnalysis(
            category=ReasonCodeCategory.BUYER_CANCELLATION,
            confidence_score=0.85,
            key_evidence=EvidenceSection(
                message_evidence=["[sep] Test message"],
                shipping_evidence=["[sep] Test shipping"],
                timeline_evidence=["[sep] Test timeline"]
            ),
            reasoning=ReasoningSection(
                primary_factors=["[sep] Test factor"],
                supporting_evidence=["[sep] Test support"],
                contradicting_evidence=["[sep] None"]
            )
        )
        
        mock_processor.agent.analyze_rnr_case_sync = Mock(return_value=mock_result)
        
        result = mock_processor.process_single_case(
            dialogue="Test dialogue",
            shiptrack="Test shiptrack",
            max_estimated_arrival_date="2025-02-22"
        )
        
        assert result['category'] == 'BuyerCancellation'
        assert result['confidence_score'] == 0.85
        assert result['processing_status'] == 'success'
        assert result['framework'] == 'PydanticAI'
        assert result['error_message'] is None
    
    def test_process_single_case_error(self, mock_processor):
        """Test single case processing with error."""
        # Mock the agent to raise an exception
        mock_processor.agent.analyze_rnr_case_sync = Mock(side_effect=Exception("Test error"))
        
        result = mock_processor.process_single_case(
            dialogue="Test dialogue",
            shiptrack="Test shiptrack",
            max_estimated_arrival_date="2025-02-22"
        )
        
        assert result['category'] == 'Insufficient_Information'
        assert result['confidence_score'] == 0.0
        assert result['processing_status'] == 'error'
        assert result['framework'] == 'PydanticAI'
        assert result['error_message'] == 'Test error'


class TestIntegration:
    """Integration tests for the complete PydanticAI implementation."""
    
    def test_end_to_end_model_flow(self):
        """Test the complete flow from input to formatted output."""
        # Create test data
        evidence = EvidenceSection(
            message_evidence=["Buyer claims package not received"],
            shipping_evidence=["Tracking shows delivered on 2025-02-21"],
            timeline_evidence=["Delivery confirmed before buyer complaint"]
        )
        
        reasoning = ReasoningSection(
            primary_factors=["Package marked as delivered but buyer disputes receipt"],
            supporting_evidence=["Clear delivery confirmation in tracking"],
            contradicting_evidence=["None"]
        )
        
        analysis = RnRReasonCodeAnalysis(
            category=ReasonCodeCategory.TRUE_DNR,
            confidence_score=0.92,
            key_evidence=evidence,
            reasoning=reasoning
        )
        
        # Test model validation
        assert analysis.category == ReasonCodeCategory.TRUE_DNR
        assert analysis.confidence_score == 0.92
        
        # Test evidence formatting (should auto-add [sep] prefix)
        assert all(item.startswith("[sep] ") for item in analysis.key_evidence.message_evidence)
        assert all(item.startswith("[sep] ") for item in analysis.key_evidence.shipping_evidence)
        assert all(item.startswith("[sep] ") for item in analysis.key_evidence.timeline_evidence)
        
        # Test reasoning formatting
        assert all(item.startswith("[sep] ") for item in analysis.reasoning.primary_factors)
        assert all(item.startswith("[sep] ") for item in analysis.reasoning.supporting_evidence)
        assert all(item.startswith("[sep] ") for item in analysis.reasoning.contradicting_evidence)
        
        # Test formatted output
        formatted = analysis.to_formatted_output()
        assert "1. Category: TrueDNR" in formatted
        assert "2. Confidence Score: 0.92" in formatted
        assert "3. Key Evidence:" in formatted
        assert "4. Reasoning:" in formatted
        
        # Verify all evidence and reasoning items are present
        assert "[sep] Buyer claims package not received" in formatted
        assert "[sep] Tracking shows delivered on 2025-02-21" in formatted
        assert "[sep] Delivery confirmed before buyer complaint" in formatted
        assert "[sep] Package marked as delivered but buyer disputes receipt" in formatted
        assert "[sep] Clear delivery confirmation in tracking" in formatted
        assert "[sep] None" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
