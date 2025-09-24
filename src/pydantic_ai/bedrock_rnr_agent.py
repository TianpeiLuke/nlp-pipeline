"""
PydanticAI-based AWS Bedrock agent for RnR Reason Code classification.
Uses PydanticAI framework with AWS Bedrock and inference profile support.
Compatible with SageMaker environments.
"""

import json
import logging
import os
from typing import Optional, Dict, Any, Union
import boto3
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.bedrock import BedrockModel

from .rnr_reason_code_models import (
    RnRReasonCodeAnalysis, 
    RnRAnalysisInput, 
    ReasonCodeCategory, 
    EvidenceSection, 
    ReasoningSection
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BedrockRnRPydanticAgent:
    """
    PydanticAI-based AWS Bedrock agent for RnR Reason Code classification.
    Uses PydanticAI's structured output capabilities with AWS Bedrock models.
    Supports inference profiles for Claude 4 and other provisioned throughput models.
    """
    
    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name: str = "us-west-2",
        max_retries: int = 3,
        inference_profile_arn: Optional[str] = None,
        use_inference_profile: Optional[bool] = None,
        **bedrock_kwargs
    ):
        """
        Initialize the PydanticAI Bedrock RnR Agent.
        
        Args:
            model_id: Bedrock model ID to use (fallback when inference profile fails)
            region_name: AWS region name
            max_retries: Maximum number of retries for API calls
            inference_profile_arn: ARN of the inference profile to use (optional)
            use_inference_profile: Whether to use inference profile (auto-detected if None)
            **bedrock_kwargs: Additional arguments for Bedrock client
        """
        # Check for environment variables
        env_inference_profile = os.getenv('BEDROCK_INFERENCE_PROFILE_ARN')
        env_use_inference_profile = os.getenv('USE_INFERENCE_PROFILE', '').lower() in ('true', '1', 'yes')
        
        # Set inference profile configuration
        self.inference_profile_arn = inference_profile_arn or env_inference_profile
        
        # Auto-detect use_inference_profile if not explicitly set
        if use_inference_profile is None:
            self.use_inference_profile = bool(self.inference_profile_arn) or env_use_inference_profile
        else:
            self.use_inference_profile = use_inference_profile
        
        # Model compatibility lists
        inference_profile_required_models = [
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "anthropic.claude-4-haiku-20250101-v1:0",
            "anthropic.claude-4-sonnet-20250101-v1:0",
            "anthropic.claude-4-opus-20250101-v1:0",
            "anthropic.claude-sonnet-4-20250514-v1:0",
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "anthropic.claude-opus-4-20250514-v1:0",
            "us.anthropic.claude-opus-4-20250514-v1:0"
        ]
        
        # Auto-detect if inference profile is needed for this model
        if model_id in inference_profile_required_models and not self.use_inference_profile:
            logger.warning(f"Model '{model_id}' typically requires inference profile. Auto-enabling inference profile mode.")
            self.use_inference_profile = True
        
        # Set fallback model for on-demand throughput
        self.fallback_model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        
        self.model_id = model_id
        self.region_name = region_name
        self.max_retries = max_retries
        
        # Initialize Bedrock client
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=region_name,
            **bedrock_kwargs
        )
        
        # Configure the model for PydanticAI
        self._configure_pydantic_model()
        
        # Initialize the PydanticAI agent
        self._initialize_agent()
        
        # Log configuration
        if self.use_inference_profile and self.inference_profile_arn:
            logger.info(f"Initialized PydanticAI Bedrock RnR Agent with inference profile: {self.inference_profile_arn} in region: {region_name}")
        elif self.use_inference_profile:
            logger.warning(f"Inference profile mode enabled but no ARN provided. Will attempt with model ID: {model_id}")
        else:
            logger.info(f"Initialized PydanticAI Bedrock RnR Agent with model: {model_id} in region: {region_name}")
    
    def _configure_pydantic_model(self):
        """Configure the PydanticAI Bedrock model."""
        # Determine which model identifier to use
        if self.use_inference_profile and self.inference_profile_arn:
            model_identifier = self.inference_profile_arn
        else:
            model_identifier = self.model_id
        
        # Create the PydanticAI Bedrock model
        self.pydantic_model = BedrockModel(
            model_id=model_identifier,
            region_name=self.region_name,
            client=self.bedrock_client
        )
    
    def _initialize_agent(self):
        """Initialize the PydanticAI agent with system prompt."""
        system_prompt = self._get_system_prompt()
        
        # Create the PydanticAI agent
        self.agent = Agent(
            model=self.pydantic_model,
            result_type=RnRReasonCodeAnalysis,
            system_prompt=system_prompt
        )
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt using the exact template provided."""
        return """You are an expert in analyzing buyer-seller messaging conversations and shipping logistics. Your task is to classify the interaction based on message content, shipping events, and delivery timing into one of several predefined categories.

Categories and their criteria:

1. TrueDNR (Delivered Not Received)
    - Package marked as delivered (EVENT_301)
    - BUT buyer claims non-receipt
    - Key elements:
        * Tracking shows delivery
        * Buyer disputes receiving
        * Delivery location discussion
        * Missing package investigation
        * Possible theft/misdelivery
    - Multi-item orders:
        * If buyer claims missing items even when other items show as delivered, classify as TrueDNR
        * Applies when any item in the order is reported missing despite delivery confirmation
        * Focus on the missing item(s) rather than successfully delivered items
      
2. Confirmed_Delay
    - Shipment delayed due to uncontrollable external factors
    - Delay confirmed by seller or shiptrack status
    - Common delay reasons:
        * Customs processing delays
        * COVID-related restrictions
        * Traffic control/accidents
        * Natural disasters/weather
        * War/political situations
        * Labor strikes
        * Carrier facility issues
    - Key indicators:
        * Seller acknowledges delay
        * Shiptrack shows delay status codes
        * External factor confirmation
        * Refund given due to confirmed delay
    - Must NOT include:
        * Unconfirmed delays
        * Buyer-only claims of delay
        * Normal transit time variations
      
3. Delivery_Attempt_Failed
    - Delivery attempt unsuccessful
    - Package returned to seller
    - Confirmed by seller or shiptrack status
    - Key indicators:
        * Failed delivery attempt events
        * Return to sender confirmation
        * Seller confirms delivery failure
        * No successful delivery scan
        * Package back in seller possession
    - Common reasons:
        * Address issues (undeliverable address)
        * Recipient unavailable
        * Access restrictions
        * Carrier unable to deliver
    - Must include seller/shiptrack confirmation
      
4. Seller_Unable_To_Ship
    - Seller offers refund directly due to shipping issues
    - Order not shipped due to seller-side problems
    - Seller-initiated refund (not buyer cancellation request)
    - Common seller issues:
        * Stock unavailable/out of stock
        * Shipping restrictions to buyer location
        * Processing problems/system issues
        * Warehouse issues/fulfillment problems
        * Carrier pickup failure
        * Inventory management errors
    - Key indicators:
        * Seller proactively contacts buyer about inability to ship
        * Seller offers refund without buyer request
        * Must occur before any shipping events
        * No shipment tracking initiated
    - Must NOT include:
        * Buyer-requested cancellations (use BuyerCancellation)
        * Cases where buyer initiates cancellation
        * Shipped items with delays (use other categories)
      
5. PDA_Undeliverable
    - Item stuck in transit without status updates or seller confirmation of reason
    - Buyer claims non-receipt while package shows shipped/in-transit
    - Seller does not admit fault or provide specific delay reason
    - Key indicators:
        * Package shows shipped/in-transit status
        * No delivery confirmation
        * No confirmed external delay factors
        * No confirmed delivery attempt failure
        * No return to sender
        * Seller cannot provide specific delay/loss reason
        * Buyer reports non-receipt
    - Covers both scenarios:
        * Package currently moving through logistics network
        * Package shipped but tracking shows no delivery or return
    - Potential abuse pattern:
        * Buyer may manipulate seller for early refund
        * Package may still be delivered after refund
    - Must NOT include:
        * Confirmed delays (use Confirmed_Delay)
        * Failed delivery attempts (use Delivery_Attempt_Failed)
        * Seller unable to ship (use Seller_Unable_To_Ship)
        * Delivered packages (use other categories)
        * Packages returned to seller
      
6. PDA_Early_Refund
    - Refund given BEFORE delivery date where:
        * Product tracking later shows successful delivery
        * No product return recorded
    - Timeline verification required:
        * Refund timestamp must precede delivery timestamp
        * Delivery must be confirmed after refund
    - Key verification:
        * Clear timestamp comparison between refund and delivery
        * No return record exists

7. Buyer_Received_WrongORDefective_Item
    - Product quality/condition issues:
        * Damaged/defective on arrival
        * Missing parts/accessories
        * Different from description
        * Wrong size/color/model
        * Functionality problems
        * Quality below expectations
        * Authenticity concerns
    - Must include buyer confirmation of receiving item
    - Usually occurs post-delivery
    - Key requirement: Eventually becomes actual return
        * Seller requests buyer to return the item
        * Buyer agrees to return before refund is given
        * Return shipping process initiated
    - Must NOT include:
        * Returnless refund scenarios (use Returnless_Refund)
        * Liquids, gels, hazardous materials
        * Fresh items (broken eggs, bad vegetables)
        * Cases where no return is expected
      
8. Returnless_Refund
    - Refund given without requiring customer to return the product
    - Clear delivery confirmation or buyer does not claim non-receipt
    - Buyer may claim received wrong or defective item but no return expected
    - Common product types qualifying for returnless refunds:
        * Liquids and gels
        * Hazardous materials (broken glass, spilled acid)
        * Fresh items (broken eggs, bad vegetables)
        * Perishable goods
        * Items unsafe to return
        * Cheap items too expensive to return (low-value products)
    - Key dialogue indicators:
        * Seller/CS agent explicitly offers refund without return
        * "This is your refund. You can keep the item."
        * "No need to return the product"
        * "Keep the item and here's your refund"
        * Explicit permission to retain the product
    - Key indicators:
        * Initial return request from buyer
        * Seller offers refund without return requirement
        * No return shipping label provided
        * No return tracking events
        * Product retention explicitly allowed
        * Cost of return exceeds item value
    - Potential abuse pattern:
        * Customers exploit returnless refund policy for free products
        * Common solution applicable to both AFN and MFN
    - Timeline verification:
        * Delivery confirmation exists
        * Return discussion occurs after delivery
        * Refund occurs after return request
        * No return shipping events follow

9. BuyerCancellation
    - Buyer cancels order for their own reasons before delivery
    - Cancellation timestamp occurs BEFORE delivery timestamp
    - Buyer does not receive the item yet (no returns involved)
    - Common reasons:
        * Change of plan/mind
        * Late delivery concerns
        * Found better alternative
        * Payment issues
        * Personal circumstances change
    - Key timing requirements:
        * Must occur before delivery
        * Must occur before seller shipment OR
        * Must occur when shiptrack shows no delay signs
    - Shiptrack status considerations:
        * If cancellation before seller shipment: BuyerCancellation
        * If cancellation while shipped/in-transit with no delay signs: use PDA_Undeliverable
        * If cancellation with confirmed delays: use Confirmed_Delay
        * If cancellation with delivery attempt failure: use Delivery_Attempt_Failed
    - Must NOT include:
        * Post-delivery scenarios (use Return_NoLongerNeeded)
        * Cases with confirmed shipping delays
        * Cases with delivery attempt failures
        * Seller-initiated refunds (use Seller_Unable_To_Ship)
      
10. Return_NoLongerNeeded
    - Post-delivery return initiation for good quality items
    - Return request timestamp occurs AFTER delivery timestamp
    - Buyer received the item but no longer needs it
    - Buyer requests return without claiming product defects or damage
    - Product received is of good quality but no longer needed by buyer
    - Common reasons:
        * Changed mind about purchase (after receiving)
        * Found better alternative (after delivery)
        * Size/fit issues (not defective)
        * Duplicate purchase realization
        * Gift not wanted
        * Personal preference change after seeing item
    - Key timing requirement:
        * Must occur AFTER delivery confirmation
        * Buyer acknowledges receiving the item
    - Key requirement: Eventually becomes actual return
        * Seller requests buyer to return the item
        * Buyer agrees to return before refund is given
        * Return shipping process initiated
    - Must NOT include:
        * Pre-delivery cancellations (use BuyerCancellation)
        * Claims of defective/damaged items (use Buyer_Received_WrongORDefective_Item)
        * Returnless refund scenarios (use Returnless_Refund)
      
11. Product_Information_Support
    - General product information and support requests
    - Not related to refund or return events
    - Documentation and information requests:
        * Invoice copies and receipts
        * Tax documents and payment records
        * Order summaries and confirmations
        * Product specifications and details
        * Warranty information
    - Pricing and promotional inquiries:
        * Coupon applications and promotional codes
        * Price matching requests
        * Volume discounts and special rates
        * Billing questions and clarifications
    - Product support and guidance:
        * Instructions on how to use the product
        * Troubleshooting assistance
        * Product customization guidance
        * Setup and installation help
        * Maintenance and care instructions
        * Compatibility questions
    - Focus on information provision, not shipping or quality issues
    - No refund or return discussion involved

12. Insufficient_Information
    - Ultimate "I don't know" category when context is missing
    - Information from dialogue and/or ship track events insufficient to understand what happens
    - Common scenarios:
        * Lack of one or both input data sources
        * Message cut off or incomplete dialogue
        * Ship track events too short or missing
        * Buyer request visible but no seller reply (no engagement)
        * Corrupted or unreadable messages
        * Non-language content or formatting issues
        * General inquiries without clear resolution
        * Non-specific customer service interactions
        * Available data insufficient for other categories
    - Use when no additional decision can be made based on available information
    - Default category when no clear classification fits
    - Indicates need for more complete data to make proper classification

CRITICAL OUTPUT REQUIREMENTS:
- Choose exactly ONE category from the provided list
- Category name must match exactly (case-sensitive)
- Provide confidence score as decimal between 0.00 and 1.00
- Structure evidence with Message Evidence, Shipping Evidence, Timeline Evidence
- Structure reasoning with Primary Factors, Supporting Evidence, Contradicting Evidence
- Each evidence/reasoning item must start with "[sep] " prefix
- Use "None" for contradicting evidence if none exists

CONFIDENCE SCORING GUIDELINES:
- Base confidence for complete data: 0.7-1.0
- Missing one field: reduce by 0.1-0.2
- Missing two fields: reduce by 0.2-0.3
- Minimum confidence threshold: 0.5
- Group A categories (abuse risk): Higher confidence requirements
- Timeline evidence critical for PDA_Early_Refund: Require 0.8+ confidence

Return a structured RnRReasonCodeAnalysis object with the classification results."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def analyze_rnr_case_async(
        self,
        dialogue: str,
        shiptrack: str,
        max_estimated_arrival_date: Optional[str] = None
    ) -> RnRReasonCodeAnalysis:
        """
        Asynchronous analysis of RnR case using PydanticAI.
        
        Args:
            dialogue: Buyer-seller conversation messages
            shiptrack: Ship track events history
            max_estimated_arrival_date: Estimated delivery date
            
        Returns:
            RnRReasonCodeAnalysis: Structured analysis result
        """
        try:
            # Create input model
            analysis_input = RnRAnalysisInput(
                dialogue=dialogue,
                shiptrack=shiptrack,
                max_estimated_arrival_date=max_estimated_arrival_date
            )
            
            # Prepare the prompt for PydanticAI
            prompt = f"""Please analyze the following RnR case using the feature-based decision tree approach:

ANALYSIS INSTRUCTIONS:
1. First determine: Does this involve refund/return or is it information-only?
2. If refund/return: Is return required or refund-only?
3. Apply Group A (refund-only) or Group B (return+refund) priority ordering
4. Verify key decision criteria (timeline evidence, return records, delivery confirmation, explicit permission)
5. Adjust confidence based on data completeness and abuse risk level

Case Data:
Dialogue: {analysis_input.dialogue}
Ship Track Events: {analysis_input.shiptrack}
Estimated Delivery: {analysis_input.max_estimated_arrival_date or 'Not provided'}

Analyze this case using the decision tree approach and return a structured RnRReasonCodeAnalysis object."""

            # Run the PydanticAI agent
            result = await self.agent.run(prompt)
            
            logger.info(f"Successfully analyzed RnR case with PydanticAI agent. Category: {result.data.category}, Confidence: {result.data.confidence_score}")
            return result.data
            
        except Exception as e:
            logger.error(f"Error analyzing RnR case with PydanticAI agent: {str(e)}")
            # Return fallback analysis instead of raising
            return self._create_fallback_analysis(f"Error: {str(e)}")

    def analyze_rnr_case_sync(
        self,
        dialogue: str,
        shiptrack: str,
        max_estimated_arrival_date: Optional[str] = None
    ) -> RnRReasonCodeAnalysis:
        """
        Synchronous wrapper for the async analysis method.
        
        Args:
            dialogue: Buyer-seller conversation messages
            shiptrack: Ship track events history
            max_estimated_arrival_date: Estimated delivery date
            
        Returns:
            RnRReasonCodeAnalysis: Structured analysis result
        """
        import asyncio
        
        try:
            # Run the async method in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.analyze_rnr_case_async(dialogue, shiptrack, max_estimated_arrival_date)
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in synchronous analysis wrapper: {str(e)}")
            return self._create_fallback_analysis(f"Sync wrapper error: {str(e)}")

    def _create_fallback_analysis(self, error_message: str) -> RnRReasonCodeAnalysis:
        """
        Create a fallback analysis when processing fails using the decision tree approach.
        
        Args:
            error_message: The error message to include
            
        Returns:
            RnRReasonCodeAnalysis with fallback values
        """
        return RnRReasonCodeAnalysis(
            category=ReasonCodeCategory.INSUFFICIENT_INFORMATION,
            confidence_score=0.0,
            key_evidence=EvidenceSection(
                message_evidence=[f"[sep] Error processing case: {error_message}"],
                shipping_evidence=["[sep] Unable to parse shipping evidence due to error"],
                timeline_evidence=["[sep] Unable to parse timeline evidence due to error"]
            ),
            reasoning=ReasoningSection(
                primary_factors=["[sep] Processing error prevented analysis"],
                supporting_evidence=[f"[sep] Technical error: {error_message}"],
                contradicting_evidence=["[sep] None"]
            )
        )

    async def batch_analyze_async(
        self,
        cases: list[Dict[str, Any]],
        batch_size: int = 10
    ) -> list[RnRReasonCodeAnalysis]:
        """
        Analyze multiple RnR cases in batches using PydanticAI.
        
        Args:
            cases: List of dictionaries containing dialogue, shiptrack, and optional max_estimated_arrival_date
            batch_size: Number of cases to process in each batch
            
        Returns:
            List of RnRReasonCodeAnalysis results
        """
        results = []
        
        for i in range(0, len(cases), batch_size):
            batch = cases[i:i + batch_size]
            batch_results = []
            
            # Process batch concurrently
            import asyncio
            tasks = []
            
            for case in batch:
                task = self.analyze_rnr_case_async(
                    dialogue=case.get('dialogue', ''),
                    shiptrack=case.get('shiptrack', ''),
                    max_estimated_arrival_date=case.get('max_estimated_arrival_date')
                )
                tasks.append(task)
            
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle any exceptions in the batch
                for idx, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing case {idx} in batch: {str(result)}")
                        batch_results[idx] = self._create_fallback_analysis(f"Batch processing error: {str(result)}")
                
            except Exception as e:
                logger.error(f"Error processing batch with PydanticAI agent: {str(e)}")
                # Create fallback results for the entire batch
                batch_results = [
                    self._create_fallback_analysis(f"Batch error: {str(e)}")
                    for _ in batch
                ]
            
            results.extend(batch_results)
            logger.info(f"PydanticAI Agent processed batch {i//batch_size + 1}/{(len(cases) + batch_size - 1)//batch_size}")
        
        return results

    def batch_analyze_sync(
        self,
        cases: list[Dict[str, Any]],
        batch_size: int = 10
    ) -> list[RnRReasonCodeAnalysis]:
        """
        Synchronous wrapper for batch analysis.
        
        Args:
            cases: List of dictionaries containing dialogue, shiptrack, and optional max_estimated_arrival_date
            batch_size: Number of cases to process in each batch
            
        Returns:
            List of RnRReasonCodeAnalysis results
        """
        import asyncio
        
        try:
            # Run the async method in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.batch_analyze_async(cases, batch_size)
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in synchronous batch analysis wrapper: {str(e)}")
            # Return fallback results for all cases
            return [
                self._create_fallback_analysis(f"Sync batch wrapper error: {str(e)}")
                for _ in cases
            ]

    # Alias for backward compatibility
    def analyze_case(self, case_data: str) -> RnRReasonCodeAnalysis:
        """
        Backward compatibility method for analyze_case.
        
        Args:
            case_data: Combined case data string
            
        Returns:
            RnRReasonCodeAnalysis: Structured analysis result
        """
        # Simple parsing of case_data string
        parts = case_data.split(" | ")
        dialogue = ""
        shiptrack = ""
        
        for part in parts:
            if part.startswith("Customer Dialogue:"):
                dialogue = part.replace("Customer Dialogue:", "").strip()
            elif part.startswith("Shipping History:"):
                shiptrack = part.replace("Shipping History:", "").strip()
        
        return self.analyze_rnr_case_sync(dialogue, shiptrack)

    def get_decision_tree_info(self) -> Dict[str, Any]:
        """
        Get information about the decision tree approach used by this PydanticAI agent.
        
        Returns:
            Dictionary containing decision tree structure and criteria
        """
        return {
            "version": "PydanticAI-2.0",
            "framework": "PydanticAI with AWS Bedrock",
            "approach": "Feature-based Category Priority Hierarchy with Structured Output",
            "decision_tree": {
                "step_1": "Determine Transaction Type (refund/return vs information-only)",
                "step_2": "Determine Return Requirement (refund-only vs return+refund)",
                "groups": {
                    "A": {
                        "name": "Refund-Only Cases",
                        "risk_level": "Highest Abuse Risk",
                        "categories": [
                            "PDA_Early_Refund",
                            "PDA_Undeliverable", 
                            "TrueDNR",
                            "Confirmed_Delay",
                            "Delivery_Attempt_Failed",
                            "Seller_Unable_To_Ship",
                            "BuyerCancellation"
                        ]
                    },
                    "B": {
                        "name": "Return + Refund Cases",
                        "risk_level": "Medium Risk",
                        "categories": [
                            "Returnless_Refund",
                            "Buyer_Received_WrongORDefective_Item",
                            "Return_NoLongerNeeded"
                        ]
                    },
                    "C": {
                        "name": "Information/Support Only",
                        "risk_level": "Lowest Risk",
                        "categories": [
                            "Product_Information_Support",
                            "Insufficient_Information"
                        ]
                    }
                }
            },
            "key_criteria": {
                "timeline_evidence": "Required for PDA_Early_Refund",
                "return_record": "Must verify absence for Group A",
                "delivery_confirmation": "Critical for TrueDNR vs PDA_Undeliverable",
                "explicit_permission": "Required for Returnless_Refund",
                "quality_claims": "Distinguishes defective vs unwanted items"
            },
            "pydantic_ai_features": {
                "structured_output": "Automatic validation and parsing",
                "type_safety": "Pydantic model validation",
                "async_support": "Native async/await support",
                "error_handling": "Robust fallback mechanisms"
            }
        }
