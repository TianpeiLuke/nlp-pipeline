
from typing import List, Optional
from pydantic import BaseModel, Field
import re


class BSMAnalysis(BaseModel):
    """Structured container for analysis results with validation"""
    category: str = Field(default="Unknown")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    message_evidence: List[str] = Field(default_factory=list)
    shipping_evidence: List[str] = Field(default_factory=list)
    timeline_evidence: List[str] = Field(default_factory=list)
    primary_factors: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)
    raw_response: str = Field(default="")
    error: Optional[str] = Field(default=None)
    latency: Optional[float] = Field(default=None)

    class Config:
        validate_assignment = True
        
        
def parse_claude_response(response: str) -> BSMAnalysis:
    """Parse Claude's response with [sep] tokens using Pydantic validation"""
    try:
        # Extract category
        category_match = re.search(r'Category:\s*([A-Za-z_]+(?:\s*\([^)]*\))?)', response)
        category = category_match.group(1) if category_match else "Unknown"

        # Extract confidence score
        confidence_match = re.search(r'Confidence Score:\s*(0?\.\d+|1\.00?)', response)
        confidence_score = float(confidence_match.group(1)) if confidence_match else 0.0

        # Helper function to extract evidence sections
        def extract_evidence(section_name: str) -> List[str]:
            # Look for section starting with * and ending before the next section
            pattern = f"\* {section_name}:.*?(?=(?:\n\s*\* |\n\d\.|\Z))"
            match = re.search(pattern, response, re.DOTALL)
            if not match:
                return []
            
            # Split by [sep] token and clean
            evidence_text = match.group(0)
            # Use positive lookbehind to ensure we get content after [sep]
            items = re.findall(r'(?<=\[sep\])(.*?)(?=(?:\[sep\]|\n\s*\*|\Z))', evidence_text, re.DOTALL)
            return [item.strip() for item in items if item.strip()]

        # Extract evidence for each section
        message_evidence = extract_evidence("Message Evidence")
        shipping_evidence = extract_evidence("Shipping Evidence")
        timeline_evidence = extract_evidence("Timeline Evidence")
        primary_factors = extract_evidence("Primary Factors")
        supporting_evidence = extract_evidence("Supporting Evidence")
        contradicting_evidence = extract_evidence("Contradicting Evidence")

        # Create BSMAnalysis instance
        analysis = BSMAnalysis(
            category=category,
            confidence_score=confidence_score,
            message_evidence=message_evidence,
            shipping_evidence=shipping_evidence,
            timeline_evidence=timeline_evidence,
            primary_factors=primary_factors,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            raw_response=response
        )

        # Validate the results
        if not analysis.category or analysis.confidence_score == 0.0:
            raise ValueError("Missing required fields: category or confidence score")

        return analysis

    except Exception as e:
        print(f"Parsing error: {str(e)}")
        print(f"Raw response: {response}")
        
        # Try to extract basic information even if full parsing fails
        try:
            category_match = re.search(r'Category:\s*([A-Za-z_]+(?:\s*\([^)]*\))?)', response)
            confidence_match = re.search(r'Confidence Score:\s*(0?\.\d+|1\.00?)', response)
            
            return BSMAnalysis(
                category=category_match.group(1) if category_match else "Error",
                confidence_score=float(confidence_match.group(1)) if confidence_match else 0.0,
                raw_response=response,
                error=f"Partial parse only: {str(e)}"
            )
        except:
            return BSMAnalysis(
                category="Error",
                confidence_score=0.0,
                raw_response=response,
                error=f"Failed to parse response: {str(e)}"
            )
        

if __name__ == 'main':
    """Test the parser with sample output"""
    test_response = """
1. Category: TrueDNR
2. Confidence Score: 0.95
3. Key Evidence:
   * Message Evidence:
     [sep] Customer message: "Package never arrived"
     [sep] Seller response: "Tracking shows delivered"
   * Shipping Evidence:
     [sep] EVENT_301: Delivered on 2024-03-15
     [sep] No subsequent delivery scans
   * Timeline Evidence:
     [sep] Delivery scan on March 15
     [sep] Customer complaint on March 16
4. Reasoning:
   * Primary Factors:
     [sep] Tracking shows delivery but customer claims non-receipt
     [sep] Timeline supports DNR claim
   * Supporting Evidence:
     [sep] Customer checked with neighbors
     [sep] No alternative delivery location found
   * Contradicting Evidence:
     [sep] None
    """
    
    result = parse_claude_response(test_response)
    print("Parsed Result:")
    print(f"Category: {result.category}")
    print(f"Confidence: {result.confidence_score}")
    print("Message Evidence:", result.message_evidence)
    print("Shipping Evidence:", result.shipping_evidence)
    print("Timeline Evidence:", result.timeline_evidencence)