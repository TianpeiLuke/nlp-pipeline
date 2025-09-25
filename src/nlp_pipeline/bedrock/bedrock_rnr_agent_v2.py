"""
Enhanced AWS Bedrock agent for RnR Reason Code classification with updated prompt.
Features improved Category Priority Hierarchy with feature-based decision tree.
Compatible with SageMaker environments.
"""

import json
import logging
import os
from typing import Optional, Dict, Any, Union
import boto3
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

from .rnr_reason_code_models import RnRReasonCodeAnalysis, RnRAnalysisInput, ReasonCodeCategory, EvidenceSection, ReasoningSection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BedrockRnRAgentV2:
    """
    Enhanced AWS Bedrock agent for RnR Reason Code classification.
    Uses boto3 bedrock-runtime directly with updated prompt featuring
    feature-based Category Priority Hierarchy.
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
        Initialize the Enhanced Bedrock RnR Agent.
        
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
        # Models that require inference profiles (provisioned throughput)
        inference_profile_required_models = [
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "anthropic.claude-4-haiku-20250101-v1:0",
            "anthropic.claude-4-sonnet-20250101-v1:0",
            "anthropic.claude-4-opus-20250101-v1:0",
            # Claude 4 models that require inference profiles
            "anthropic.claude-sonnet-4-20250514-v1:0",
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "anthropic.claude-opus-4-20250514-v1:0",
            "us.anthropic.claude-opus-4-20250514-v1:0"
        ]
        
        # Models compatible with on-demand throughput
        on_demand_compatible_models = [
            # Claude 2 models (legacy)
            "anthropic.claude-v2",
            "anthropic.claude-v2:1",
            # Claude 3 models
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0", 
            "anthropic.claude-3-opus-20240229-v1:0",
            # Claude 3.5 models (older versions)
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-3-5-haiku-20241022-v1:0",
            # Claude Instant (legacy)
            "anthropic.claude-instant-v1"
        ]
        
        # Auto-detect if inference profile is needed for this model
        if model_id in inference_profile_required_models and not self.use_inference_profile:
            logger.warning(f"Model '{model_id}' typically requires inference profile. Auto-enabling inference profile mode.")
            self.use_inference_profile = True
        
        # Set fallback model for on-demand throughput
        self.fallback_model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"  # Known compatible model
        
        self.model_id = model_id
        self.region_name = region_name
        self.max_retries = max_retries
        
        # Initialize Bedrock client
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=region_name,
            **bedrock_kwargs
        )
        
        # Log configuration
        if self.use_inference_profile and self.inference_profile_arn:
            logger.info(f"Initialized Enhanced Bedrock RnR Agent V2 with inference profile: {self.inference_profile_arn} in region: {region_name}")
        elif self.use_inference_profile:
            logger.warning(f"Inference profile mode enabled but no ARN provided. Will attempt with model ID: {model_id}")
        else:
            logger.info(f"Initialized Enhanced Bedrock RnR Agent V2 with model: {model_id} in region: {region_name}")
    
    def _get_system_prompt(self) -> str:
        """Get the enhanced system prompt with feature-based Category Priority Hierarchy."""
        return """You are an expert in analyzing buyer-seller messaging conversations and shipping logistics. Your task is to classify the interaction based on message content, shipping events, and delivery timing into one of several predefined categories using a feature-based decision tree approach.

CLASSIFICATION DECISION TREE - Apply in Order:

Step 1: Determine Transaction Type
- Does the case involve refund or return? → Continue to Step 2
- No financial transaction (information only)? → Product_Information_Support
- Cannot determine from available data? → Insufficient_Information

Step 2: Determine Return Requirement
- Refund granted WITHOUT return expected? → Group A: Refund-Only Cases
- Refund granted WITH return required? → Group B: Return + Refund Cases

GROUP A: REFUND-ONLY CASES (Highest Abuse Risk)
Priority Order within Group A:
1. PDA_Early_Refund - Refund before delivery (verify timeline evidence)
2. PDA_Undeliverable - Item lost/stuck, no delivery confirmation
3. TrueDNR - Delivered but buyer disputes receipt
4. Confirmed_Delay - External delay factors confirmed
5. Delivery_Attempt_Failed - Failed delivery, returned to seller
6. Seller_Unable_To_Ship - Seller cannot fulfill order
7. BuyerCancellation - Pre-delivery buyer cancellation

GROUP B: RETURN + REFUND CASES (Medium Risk)
Priority Order within Group B:
1. Returnless_Refund - Exception case: refund without return (verify explicit permission)
2. Buyer_Received_WrongORDefective_Item - Quality issues requiring return
3. Return_NoLongerNeeded - Post-delivery unwanted item return

GROUP C: INFORMATION/SUPPORT ONLY (Lowest Risk)
1. Product_Information_Support - Information requests, no financial impact
2. Insufficient_Information - Missing context for classification

KEY DECISION CRITERIA:
- Timeline Evidence: Required for PDA_Early_Refund (refund timestamp < delivery timestamp)
- Return Record: Must verify absence for Group A categories
- Delivery Confirmation: Critical for TrueDNR vs PDA_Undeliverable distinction
- Explicit Permission: Required for Returnless_Refund (seller states "keep the item")
- Quality Claims: Distinguishes defective items from unwanted items

DETAILED CATEGORY DEFINITIONS:

1. TrueDNR (Delivered Not Received)
- Package marked as delivered (EVENT_301) BUT buyer claims non-receipt
- Key elements: Tracking shows delivery, Buyer disputes receiving, Delivery location discussion, Missing package investigation
- Multi-item orders: If buyer claims missing items even when other items show as delivered, classify as TrueDNR

2. Confirmed_Delay
- Shipment delayed due to uncontrollable external factors
- Delay confirmed by seller or shiptrack status
- Common delay reasons: Customs processing, COVID restrictions, weather, strikes, accidents, war, carrier facility issues
- Key indicators: Seller acknowledges delay, Shiptrack shows delay status codes, External factor confirmation

3. Delivery_Attempt_Failed
- Delivery attempt unsuccessful, Package returned to seller
- Confirmed by seller or shiptrack status
- Key indicators: Failed delivery attempt events, Return to sender confirmation, No successful delivery scan
- Common reasons: Address issues, Recipient unavailable, Access restrictions

4. Seller_Unable_To_Ship
- Seller offers refund directly due to shipping issues
- Order not shipped due to seller-side problems, Seller-initiated refund (not buyer cancellation)
- Common issues: Stock unavailable, Shipping restrictions, Processing problems, Warehouse issues
- Key indicators: Seller proactively contacts buyer, Must occur before shipping events

5. PDA_Undeliverable
- Item stuck in transit without status updates or seller confirmation
- Buyer claims non-receipt while package shows shipped/in-transit
- Seller cannot provide specific delay/loss reason
- Potential abuse pattern: Buyer may manipulate seller for early refund

6. PDA_Early_Refund
- Refund given BEFORE delivery date where product tracking later shows successful delivery
- Timeline verification required: Refund timestamp must precede delivery timestamp
- No product return recorded

7. Buyer_Received_WrongORDefective_Item
- Product quality/condition issues: Damaged/defective, Missing parts, Wrong item, Functionality problems
- Must include buyer confirmation of receiving item
- Key requirement: Eventually becomes actual return (seller requests return, buyer agrees)

8. Returnless_Refund
- Refund given without requiring customer to return the product
- Clear delivery confirmation or buyer does not claim non-receipt
- Key dialogue indicators: "Keep the item", "No need to return", Explicit permission to retain product
- Common product types: Liquids, hazardous materials, perishables, low-value items

9. BuyerCancellation
- Buyer cancels order for their own reasons before delivery
- Cancellation timestamp occurs BEFORE delivery timestamp
- Common reasons: Change of mind, Late delivery concerns, Payment issues
- Key timing: Must occur before delivery or before seller shipment

10. Return_NoLongerNeeded
- Post-delivery return initiation for good quality items
- Return request timestamp occurs AFTER delivery timestamp
- Buyer received item but no longer needs it
- Common reasons: Changed mind after receiving, Size issues, Duplicate purchase

11. Product_Information_Support
- General product information and support requests
- Not related to refund or return events
- Documentation requests: Invoices, receipts, specifications, warranty information
- Product support: Usage instructions, troubleshooting, compatibility questions

12. Insufficient_Information
- Ultimate "I don't know" category when context is missing
- Information insufficient to understand what happens
- Common scenarios: Incomplete dialogue, Missing shiptrack, No seller engagement, Corrupted messages

CRITICAL OUTPUT REQUIREMENTS:
- Choose exactly ONE category from the provided list
- Category name must match exactly (case-sensitive)
- Provide confidence score as decimal between 0.00 and 1.00
- Structure evidence with Message Evidence, Shipping Evidence, Timeline Evidence
- Structure reasoning with Primary Factors, Supporting Evidence, Contradicting Evidence
- Each evidence/reasoning item must start with "[sep] " prefix
- Use "None" for contradicting evidence if none exists
- Follow the exact JSON structure for the Pydantic model

CONFIDENCE SCORING GUIDELINES:
- Base confidence for complete data: 0.7-1.0
- Missing one field: reduce by 0.1-0.2
- Missing two fields: reduce by 0.2-0.3
- Minimum confidence threshold: 0.5
- Group A categories (abuse risk): Higher confidence requirements
- Timeline evidence critical for PDA_Early_Refund: Require 0.8+ confidence

Analyze the provided dialogue, shiptrack events, and estimated delivery date using the decision tree approach."""

    def _call_bedrock_api(self, prompt: str) -> Dict[str, Any]:
        """
        Call Bedrock API directly using boto3 with inference profile support and fallback logic.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            Dictionary containing the model response
        """
        # Prepare the request body for Claude models
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4000,
            "system": self._get_system_prompt(),
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "top_p": 0.9
        }
        
        # Determine which model identifier to use
        if self.use_inference_profile and self.inference_profile_arn:
            model_identifier = self.inference_profile_arn
            logger.debug(f"Using inference profile: {model_identifier}")
        else:
            model_identifier = self.model_id
            logger.debug(f"Using model ID: {model_identifier}")
        
        try:
            # Primary attempt with configured identifier
            response = self.bedrock_client.invoke_model(
                modelId=model_identifier,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json"
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            return response_body
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_message = str(e)
            
            # Check if this is an inference profile validation error
            if 'ValidationException' in error_code and 'inference profile' in error_message.lower():
                logger.warning(f"Inference profile error with {model_identifier}: {error_message}")
                
                # Try fallback strategy
                if self.use_inference_profile:
                    logger.info(f"Falling back to on-demand model: {self.fallback_model_id}")
                    try:
                        fallback_response = self.bedrock_client.invoke_model(
                            modelId=self.fallback_model_id,
                            body=json.dumps(request_body),
                            contentType="application/json",
                            accept="application/json"
                        )
                        
                        response_body = json.loads(fallback_response['body'].read())
                        logger.info("Successfully used fallback model")
                        return response_body
                        
                    except Exception as fallback_error:
                        logger.error(f"Fallback model also failed: {fallback_error}")
                        raise e  # Raise original error
                else:
                    logger.error(f"Model validation error: {error_message}")
                    raise e
            else:
                logger.error(f"Bedrock API error: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Unexpected error calling Bedrock: {e}")
            raise

    def _parse_response_to_analysis(self, response_text: str) -> RnRReasonCodeAnalysis:
        """
        Parse the model response into a structured RnRReasonCodeAnalysis object.
        
        Args:
            response_text: Raw response text from the model
            
        Returns:
            RnRReasonCodeAnalysis object
        """
        try:
            # Try to extract JSON from the response
            # Look for JSON-like structure in the response
            import re
            
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_json = json.loads(json_str)
                
                # Create RnRReasonCodeAnalysis from parsed JSON
                return RnRReasonCodeAnalysis(**parsed_json)
            
            # If no JSON found, create a fallback analysis
            logger.warning("No valid JSON found in response, creating fallback analysis")
            return self._create_fallback_analysis(response_text)
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse response as JSON: {e}")
            return self._create_fallback_analysis(response_text)

    def _create_fallback_analysis(self, response_text: str) -> RnRReasonCodeAnalysis:
        """
        Create a fallback analysis when parsing fails using the decision tree approach.
        
        Args:
            response_text: The original response text
            
        Returns:
            RnRReasonCodeAnalysis with fallback values
        """
        # Enhanced keyword-based classification following decision tree
        text_lower = response_text.lower()
        
        # Step 1: Check for information-only cases
        if any(keyword in text_lower for keyword in ["invoice", "receipt", "specification", "instruction", "troubleshoot"]):
            category = ReasonCodeCategory.PRODUCT_INFORMATION_SUPPORT
            confidence = 0.6
        # Step 2: Group A - Refund-only cases (highest priority)
        elif "early refund" in text_lower or ("refund" in text_lower and "before delivery" in text_lower):
            category = ReasonCodeCategory.PDA_EARLY_REFUND
            confidence = 0.7
        elif "never received" in text_lower or "not delivered" in text_lower:
            if "delivered" in text_lower:
                category = ReasonCodeCategory.TRUE_DNR
            else:
                category = ReasonCodeCategory.PDA_UNDELIVERABLE
            confidence = 0.6
        elif "late" in text_lower or "delayed" in text_lower:
            category = ReasonCodeCategory.CONFIRMED_DELAY
            confidence = 0.6
        elif "failed delivery" in text_lower or "delivery attempt" in text_lower:
            category = ReasonCodeCategory.DELIVERY_ATTEMPT_FAILED
            confidence = 0.6
        elif "unable to ship" in text_lower or "out of stock" in text_lower:
            category = ReasonCodeCategory.SELLER_UNABLE_TO_SHIP
            confidence = 0.6
        elif "cancel" in text_lower and "before" in text_lower:
            category = ReasonCodeCategory.BUYER_CANCELLATION
            confidence = 0.6
        # Step 3: Group B - Return + refund cases
        elif "keep the item" in text_lower or "no need to return" in text_lower:
            category = ReasonCodeCategory.RETURNLESS_REFUND
            confidence = 0.7
        elif "damaged" in text_lower or "defective" in text_lower or "wrong item" in text_lower:
            category = ReasonCodeCategory.BUYER_RECEIVED_WRONG_OR_DEFECTIVE_ITEM
            confidence = 0.6
        elif "no longer needed" in text_lower or "changed mind" in text_lower:
            category = ReasonCodeCategory.RETURN_NO_LONGER_NEEDED
            confidence = 0.6
        else:
            category = ReasonCodeCategory.INSUFFICIENT_INFORMATION
            confidence = 0.5
        
        return RnRReasonCodeAnalysis(
            category=category,
            confidence_score=confidence,
            key_evidence=EvidenceSection(
                message_evidence=["[sep] Fallback analysis based on enhanced decision tree keywords"],
                shipping_evidence=["[sep] Unable to parse detailed shipping evidence"],
                timeline_evidence=["[sep] Unable to parse detailed timeline evidence"]
            ),
            reasoning=ReasoningSection(
                primary_factors=["[sep] Fallback classification using feature-based decision tree"],
                supporting_evidence=[f"[sep] Classified as {category.value} based on enhanced keyword analysis"],
                contradicting_evidence=["[sep] None"]
            )
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def analyze_rnr_case_sync(
        self,
        dialogue: str,
        shiptrack: str,
        max_estimated_arrival_date: Optional[str] = None
    ) -> RnRReasonCodeAnalysis:
        """
        Synchronous analysis of RnR case using enhanced decision tree approach.
        
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
            
            # Prepare the enhanced prompt with decision tree guidance
            prompt = f"""Please analyze the following RnR case using the feature-based decision tree approach and return your response as a valid JSON object matching this exact structure:

{{
    "category": "one of: TrueDNR, Confirmed_Delay, Delivery_Attempt_Failed, Seller_Unable_To_Ship, PDA_Undeliverable, PDA_Early_Refund, Buyer_Received_WrongORDefective_Item, Returnless_Refund, BuyerCancellation, Return_NoLongerNeeded, Product_Information_Support, Insufficient_Information",
    "confidence_score": 0.85,
    "key_evidence": {{
        "message_evidence": ["[sep] Evidence from dialogue"],
        "shipping_evidence": ["[sep] Evidence from shipping"],
        "timeline_evidence": ["[sep] Evidence from timeline"]
    }},
    "reasoning": {{
        "primary_factors": ["[sep] Main reasoning factors following decision tree"],
        "supporting_evidence": ["[sep] Supporting evidence"],
        "contradicting_evidence": ["[sep] Contradicting evidence or None"]
    }}
}}

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

Analyze this case using the decision tree approach and respond with ONLY the JSON object, no additional text."""

            # Call Bedrock API
            response = self._call_bedrock_api(prompt)
            
            # Extract response text
            if 'content' in response and len(response['content']) > 0:
                response_text = response['content'][0].get('text', '')
            else:
                raise ValueError("No content in Bedrock response")
            
            # Parse response to structured analysis
            result = self._parse_response_to_analysis(response_text)
            
            logger.info(f"Successfully analyzed RnR case with V2 agent. Category: {result.category}, Confidence: {result.confidence_score}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing RnR case with V2 agent: {str(e)}")
            # Return fallback analysis instead of raising
            return self._create_fallback_analysis(f"Error: {str(e)}")

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

    def batch_analyze(
        self,
        cases: list[Dict[str, Any]],
        batch_size: int = 10
    ) -> list[RnRReasonCodeAnalysis]:
        """
        Analyze multiple RnR cases in batches using enhanced decision tree approach.
        
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
            
            for case in batch:
                try:
                    result = self.analyze_rnr_case_sync(
                        dialogue=case.get('dialogue', ''),
                        shiptrack=case.get('shiptrack', ''),
                        max_estimated_arrival_date=case.get('max_estimated_arrival_date')
                    )
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing case in batch with V2 agent: {str(e)}")
                    # Create a fallback result for failed cases
                    fallback_result = RnRReasonCodeAnalysis(
                        category="Insufficient_Information",
                        confidence_score=0.0,
                        key_evidence={
                            "message_evidence": ["[sep] Error processing case"],
                            "shipping_evidence": ["[sep] Error processing case"],
                            "timeline_evidence": ["[sep] Error processing case"]
                        },
                        reasoning={
                            "primary_factors": ["[sep] Processing error occurred"],
                            "supporting_evidence": ["[sep] Unable to analyze due to technical error"],
                            "contradicting_evidence": ["[sep] None"]
                        }
                    )
                    batch_results.append(fallback_result)
            
            results.extend(batch_results)
            logger.info(f"V2 Agent processed batch {i//batch_size + 1}/{(len(cases) + batch_size - 1)//batch_size}")
        
        return results

    def get_decision_tree_info(self) -> Dict[str, Any]:
        """
        Get information about the decision tree approach used by this agent.
        
        Returns:
            Dictionary containing decision tree structure and criteria
        """
        return {
            "version": "2.0",
            "approach": "Feature-based Category Priority Hierarchy",
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
            }
        }
