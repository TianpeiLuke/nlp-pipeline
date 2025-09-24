"""
Pydantic models for RnR Reason Code classification using PydanticAI.
Based on the original models but optimized for PydanticAI framework.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ReasonCodeCategory(str, Enum):
    """Enumeration of all possible reason code categories."""
    TRUE_DNR = "TrueDNR"
    CONFIRMED_DELAY = "Confirmed_Delay"
    DELIVERY_ATTEMPT_FAILED = "Delivery_Attempt_Failed"
    SELLER_UNABLE_TO_SHIP = "Seller_Unable_To_Ship"
    PDA_UNDELIVERABLE = "PDA_Undeliverable"
    PDA_EARLY_REFUND = "PDA_Early_Refund"
    BUYER_RECEIVED_WRONG_OR_DEFECTIVE_ITEM = "Buyer_Received_WrongORDefective_Item"
    RETURNLESS_REFUND = "Returnless_Refund"
    BUYER_CANCELLATION = "BuyerCancellation"
    RETURN_NO_LONGER_NEEDED = "Return_NoLongerNeeded"
    PRODUCT_INFORMATION_SUPPORT = "Product_Information_Support"
    INSUFFICIENT_INFORMATION = "Insufficient_Information"


class EvidenceSection(BaseModel):
    """Model for evidence sections in the analysis."""
    message_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence from message content with [sep] prefix"
    )
    shipping_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence from shipping events with [sep] prefix"
    )
    timeline_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence from timeline analysis with [sep] prefix"
    )

    @field_validator('message_evidence', 'shipping_evidence', 'timeline_evidence', mode='before')
    @classmethod
    def validate_evidence_format(cls, v):
        """Ensure evidence items start with [sep] prefix."""
        if isinstance(v, list):
            formatted_evidence = []
            for item in v:
                if isinstance(item, str):
                    if not item.startswith("[sep] "):
                        formatted_evidence.append(f"[sep] {item}")
                    else:
                        formatted_evidence.append(item)
                else:
                    formatted_evidence.append(f"[sep] {str(item)}")
            return formatted_evidence
        return v


class ReasoningSection(BaseModel):
    """Model for reasoning sections in the analysis."""
    primary_factors: List[str] = Field(
        default_factory=list,
        description="Main reasons for classification with [sep] prefix"
    )
    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="Additional supporting evidence with [sep] prefix"
    )
    contradicting_evidence: List[str] = Field(
        default_factory=list,
        description="Contradicting evidence or 'None' with [sep] prefix"
    )

    @field_validator('primary_factors', 'supporting_evidence', 'contradicting_evidence', mode='before')
    @classmethod
    def validate_reasoning_format(cls, v):
        """Ensure reasoning items start with [sep] prefix."""
        if isinstance(v, list):
            formatted_reasoning = []
            for item in v:
                if isinstance(item, str):
                    if not item.startswith("[sep] "):
                        formatted_reasoning.append(f"[sep] {item}")
                    else:
                        formatted_reasoning.append(item)
                else:
                    formatted_reasoning.append(f"[sep] {str(item)}")
            return formatted_reasoning
        return v


class RnRReasonCodeAnalysis(BaseModel):
    """
    Main model for RnR Reason Code analysis output.
    Follows the exact format specified in the prompt template.
    Optimized for PydanticAI structured output.
    """
    category: ReasonCodeCategory = Field(
        description="Exactly one category from the predefined list"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0.00 and 1.00"
    )
    key_evidence: EvidenceSection = Field(
        description="Structured evidence supporting the classification"
    )
    reasoning: ReasoningSection = Field(
        description="Detailed reasoning for the classification"
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True
    )

    @field_validator('confidence_score')
    @classmethod
    def validate_confidence_score(cls, v):
        """Ensure confidence score is properly formatted."""
        return round(v, 2)

    def to_formatted_output(self) -> str:
        """
        Convert the analysis to the exact format required by the prompt template.
        """
        output_lines = [
            f"1. Category: {self.category.value}",
            "",
            f"2. Confidence Score: {self.confidence_score:.2f}",
            "",
            "3. Key Evidence:",
            "   * Message Evidence:"
        ]
        
        # Add message evidence
        for evidence in self.key_evidence.message_evidence:
            output_lines.append(f"     {evidence}")
        
        output_lines.append("   * Shipping Evidence:")
        # Add shipping evidence
        for evidence in self.key_evidence.shipping_evidence:
            output_lines.append(f"     {evidence}")
        
        output_lines.append("   * Timeline Evidence:")
        # Add timeline evidence
        for evidence in self.key_evidence.timeline_evidence:
            output_lines.append(f"     {evidence}")
        
        output_lines.extend([
            "",
            "4. Reasoning:",
            "   * Primary Factors:"
        ])
        
        # Add primary factors
        for factor in self.reasoning.primary_factors:
            output_lines.append(f"     {factor}")
        
        output_lines.append("   * Supporting Evidence:")
        # Add supporting evidence
        for evidence in self.reasoning.supporting_evidence:
            output_lines.append(f"     {evidence}")
        
        output_lines.append("   * Contradicting Evidence:")
        # Add contradicting evidence
        if not self.reasoning.contradicting_evidence:
            output_lines.append("     [sep] None")
        else:
            for evidence in self.reasoning.contradicting_evidence:
                output_lines.append(f"     {evidence}")
        
        return "\n".join(output_lines)


class RnRAnalysisInput(BaseModel):
    """Input model for the RnR analysis."""
    dialogue: str = Field(description="Buyer-seller conversation messages")
    shiptrack: str = Field(description="Ship track events history")
    max_estimated_arrival_date: Optional[str] = Field(
        default=None,
        description="Estimated delivery date"
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )


class RnRCaseData(BaseModel):
    """Extended input model for batch processing with additional metadata."""
    dialogue: str = Field(description="Buyer-seller conversation messages")
    shiptrack: str = Field(description="Ship track events history")
    max_estimated_arrival_date: Optional[str] = Field(
        default=None,
        description="Estimated delivery date"
    )
    case_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the case"
    )
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="Additional metadata for the case"
    )

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for flexibility
        validate_assignment=True
    )
