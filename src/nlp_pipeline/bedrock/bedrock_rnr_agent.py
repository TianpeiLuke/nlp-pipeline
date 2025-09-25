"""
Direct AWS Bedrock agent for RnR Reason Code classification without pydantic-ai dependency.
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


class BedrockRnRAgent:
    """
    Direct AWS Bedrock agent for RnR Reason Code classification.
    Uses boto3 bedrock-runtime directly without pydantic-ai dependency.
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
        Initialize the Bedrock RnR Agent.
        
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
            logger.info(f"Initialized Bedrock RnR Agent with inference profile: {self.inference_profile_arn} in region: {region_name}")
        elif self.use_inference_profile:
            logger.warning(f"Inference profile mode enabled but no ARN provided. Will attempt with model ID: {model_id}")
        else:
            logger.info(f"Initialized Bedrock RnR Agent with model: {model_id} in region: {region_name}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the RnR classification task."""
        return """You are an expert in analyzing buyer-seller messaging conversations and shipping logistics. Your task is to classify the interaction based on message content, shipping events, and delivery timing into one of several predefined categories.

Categories and their criteria:

1. TrueDNR (Delivered Not Received)
    - Package marked as delivered (EVENT_301)
    - BUT buyer claims non-receipt
    - Key elements: Tracking shows delivery, Buyer disputes receiving, Delivery location discussion, Missing package investigation, Possible theft/misdelivery

2. Confirmed_Delay
    - Shipment delayed due to uncontrollable external factors
    - Delay confirmed by seller or shiptrack status
    - Common delay reasons: Customs processing delays, COVID-related restrictions, Traffic control/accidents, Natural disasters/weather, War/political situations, Labor strikes, Carrier facility issues
    - Key indicators: Seller acknowledges delay, Shiptrack shows delay status codes, External factor confirmation, Refund given due to confirmed delay
    - Must NOT include: Unconfirmed delays, Buyer-only claims of delay, Normal transit time variations

3. Delivery_Attempt_Failed
    - Delivery attempt unsuccessful, Package returned to seller
    - Confirmed by seller or shiptrack status
    - Key indicators: Failed delivery attempt events, Return to sender confirmation, Seller confirms delivery failure, No successful delivery scan, Package back in seller possession
    - Common reasons: Address issues (undeliverable address), Recipient unavailable, Access restrictions, Carrier unable to deliver
    - Must include seller/shiptrack confirmation

4. Seller_Unable_To_Ship
    - Seller offers refund directly due to shipping issues
    - Order not shipped due to seller-side problems
    - Seller-initiated refund (not buyer cancellation request)
    - Common seller issues: Stock unavailable/out of stock, Shipping restrictions to buyer location, Processing problems/system issues, Warehouse issues/fulfillment problems, Carrier pickup failure, Inventory management errors
    - Key indicators: Seller proactively contacts buyer about inability to ship, Seller offers refund without buyer request, Must occur before any shipping events, No shipment tracking initiated
    - Must NOT include: Buyer-requested cancellations (use BuyerCancellation), Cases where buyer initiates cancellation, Shipped items with delays (use other categories)

5. PDA_Undeliverable
    - Item stuck in transit without status updates or seller confirmation of reason
    - Buyer claims non-receipt while package shows shipped/in-transit
    - Seller does not admit fault or provide specific delay reason
    - Key indicators: Package shows shipped/in-transit status, No delivery confirmation, No confirmed external delay factors, No confirmed delivery attempt failure, No return to sender, Seller cannot provide specific delay/loss reason, Buyer reports non-receipt
    - Covers both scenarios: Package currently moving through logistics network, Package shipped but tracking shows no delivery or return
    - Potential abuse pattern: Buyer may manipulate seller for early refund, Package may still be delivered after refund
    - Must NOT include: Confirmed delays (use Confirmed_Delay), Failed delivery attempts (use Delivery_Attempt_Failed), Seller unable to ship (use Seller_Unable_To_Ship), Delivered packages (use other categories), Packages returned to seller

6. PDA_Early_Refund
    - Refund given BEFORE delivery date where: Product tracking later shows successful delivery, No product return recorded
    - Timeline verification required: Refund timestamp must precede delivery timestamp, Delivery must be confirmed after refund
    - Key verification: Clear timestamp comparison between refund and delivery, No return record exists

7. Buyer_Received_WrongORDefective_Item
    - Product quality/condition issues: Damaged/defective on arrival, Missing parts/accessories, Different from description, Wrong size/color/model, Functionality problems, Quality below expectations, Authenticity concerns
    - Must include buyer confirmation of receiving item
    - Usually occurs post-delivery
    - Key requirement: Eventually becomes actual return: Seller requests buyer to return the item, Buyer agrees to return before refund is given, Return shipping process initiated
    - Must NOT include: Returnless refund scenarios (use Returnless_Refund), Liquids, gels, hazardous materials, Fresh items (broken eggs, bad vegetables), Cases where no return is expected

8. Returnless_Refund
    - Refund given without requiring customer to return the product
    - Clear delivery confirmation or buyer does not claim non-receipt
    - Buyer may claim received wrong or defective item but no return expected
    - Common product types qualifying for returnless refunds: Liquids and gels, Hazardous materials (broken glass, spilled acid), Fresh items (broken eggs, bad vegetables), Perishable goods, Items unsafe to return, Cheap items too expensive to return (low-value products)
    - Key dialogue indicators: Seller/CS agent explicitly offers refund without return, "This is your refund. You can keep the item.", "No need to return the product", "Keep the item and here's your refund", Explicit permission to retain the product
    - Key indicators: Initial return request from buyer, Seller offers refund without return requirement, No return shipping label provided, No return tracking events, Product retention explicitly allowed, Cost of return exceeds item value
    - Potential abuse pattern: Customers exploit returnless refund policy for free products, Common solution applicable to both AFN and MFN
    - Timeline verification: Delivery confirmation exists, Return discussion occurs after delivery, Refund occurs after return request, No return shipping events follow

9. BuyerCancellation
    - Buyer cancels order for their own reasons before delivery
    - Cancellation timestamp occurs BEFORE delivery timestamp
    - Buyer does not receive the item yet (no returns involved)
    - Common reasons: Change of plan/mind, Late delivery concerns, Found better alternative, Payment issues, Personal circumstances change
    - Key timing requirements: Must occur before delivery, Must occur before seller shipment OR Must occur when shiptrack shows no delay signs
    - Shiptrack status considerations: If cancellation before seller shipment: BuyerCancellation, If cancellation while shipped/in-transit with no delay signs: use PDA_Undeliverable, If cancellation with confirmed delays: use Confirmed_Delay, If cancellation with delivery attempt failure: use Delivery_Attempt_Failed
    - Must NOT include: Post-delivery scenarios (use Return_NoLongerNeeded), Cases with confirmed shipping delays, Cases with delivery attempt failures, Seller-initiated refunds (use Seller_Unable_To_Ship)

10. Return_NoLongerNeeded
    - Post-delivery return initiation for good quality items
    - Return request timestamp occurs AFTER delivery timestamp
    - Buyer received the item but no longer needs it
    - Buyer requests return without claiming product defects or damage
    - Product received is of good quality but no longer needed by buyer
    - Common reasons: Changed mind about purchase (after receiving), Found better alternative (after delivery), Size/fit issues (not defective), Duplicate purchase realization, Gift not wanted, Personal preference change after seeing item
    - Key timing requirement: Must occur AFTER delivery confirmation, Buyer acknowledges receiving the item
    - Key requirement: Eventually becomes actual return: Seller requests buyer to return the item, Buyer agrees to return before refund is given, Return shipping process initiated
    - Must NOT include: Pre-delivery cancellations (use BuyerCancellation), Claims of defective/damaged items (use Buyer_Received_WrongORDefective_Item), Returnless refund scenarios (use Returnless_Refund)

11. Product_Information_Support
    - General product information and support requests
    - Not related to refund or return events
    - Documentation and information requests: Invoice copies and receipts, Tax documents and payment records, Order summaries and confirmations, Product specifications and details, Warranty information
    - Pricing and promotional inquiries: Coupon applications and promotional codes, Price matching requests, Volume discounts and special rates, Billing questions and clarifications
    - Product support and guidance: Instructions on how to use the product, Troubleshooting assistance, Product customization guidance, Setup and installation help, Maintenance and care instructions, Compatibility questions
    - Focus on information provision, not shipping or quality issues
    - No refund or return discussion involved

12. Insufficient_Information
    - Ultimate "I don't know" category when context is missing
    - Information from dialogue and/or ship track events insufficient to understand what happens
    - Common scenarios: Lack of one or both input data sources, Message cut off or incomplete dialogue, Ship track events too short or missing, Buyer request visible but no seller reply (no engagement), Corrupted or unreadable messages, Non-language content or formatting issues, General inquiries without clear resolution, Non-specific customer service interactions, Available data insufficient for other categories
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
- Follow the exact JSON structure for the Pydantic model

Analyze the provided dialogue, shiptrack events, and estimated delivery date to classify the interaction."""

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
        Create a fallback analysis when parsing fails.
        
        Args:
            response_text: The original response text
            
        Returns:
            RnRReasonCodeAnalysis with fallback values
        """
        # Simple keyword-based classification as fallback
        text_lower = response_text.lower()
        
        if "never received" in text_lower or "not delivered" in text_lower:
            category = ReasonCodeCategory.TRUE_DNR
            confidence = 0.6
        elif "late" in text_lower or "delayed" in text_lower:
            category = ReasonCodeCategory.CONFIRMED_DELAY
            confidence = 0.6
        elif "damaged" in text_lower or "defective" in text_lower:
            category = ReasonCodeCategory.BUYER_RECEIVED_WRONG_OR_DEFECTIVE_ITEM
            confidence = 0.6
        elif "cancel" in text_lower:
            category = ReasonCodeCategory.BUYER_CANCELLATION
            confidence = 0.6
        else:
            category = ReasonCodeCategory.INSUFFICIENT_INFORMATION
            confidence = 0.5
        
        return RnRReasonCodeAnalysis(
            category=category,
            confidence_score=confidence,
            key_evidence=EvidenceSection(
                message_evidence=["[sep] Fallback analysis based on keywords"],
                shipping_evidence=["[sep] Unable to parse detailed shipping evidence"],
                timeline_evidence=["[sep] Unable to parse detailed timeline evidence"]
            ),
            reasoning=ReasoningSection(
                primary_factors=["[sep] Fallback classification due to parsing error"],
                supporting_evidence=[f"[sep] Classified as {category.value} based on keyword analysis"],
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
        Synchronous version of analyze_rnr_case.
        
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
            
            # Prepare the prompt with JSON schema request
            prompt = f"""Please analyze the following RnR case and return your response as a valid JSON object matching this exact structure:

{{
    "category": "one of: TrueDNR, Confirmed_Delay, Delivery_Attempt_Failed, Seller_Unable_To_Ship, PDA_Undeliverable, PDA_Early_Refund, Buyer_Received_WrongORDefective_Item, Returnless_Refund, BuyerCancellation, Return_NoLongerNeeded, Product_Information_Support, Insufficient_Information",
    "confidence_score": 0.85,
    "key_evidence": {{
        "message_evidence": ["[sep] Evidence from dialogue"],
        "shipping_evidence": ["[sep] Evidence from shipping"],
        "timeline_evidence": ["[sep] Evidence from timeline"]
    }},
    "reasoning": {{
        "primary_factors": ["[sep] Main reasoning factors"],
        "supporting_evidence": ["[sep] Supporting evidence"],
        "contradicting_evidence": ["[sep] Contradicting evidence or None"]
    }}
}}

Case Data:
Dialogue: {analysis_input.dialogue}
Ship Track Events: {analysis_input.shiptrack}
Estimated Delivery: {analysis_input.max_estimated_arrival_date or 'Not provided'}

Analyze this case and respond with ONLY the JSON object, no additional text."""

            # Call Bedrock API
            response = self._call_bedrock_api(prompt)
            
            # Extract response text
            if 'content' in response and len(response['content']) > 0:
                response_text = response['content'][0].get('text', '')
            else:
                raise ValueError("No content in Bedrock response")
            
            # Parse response to structured analysis
            result = self._parse_response_to_analysis(response_text)
            
            logger.info(f"Successfully analyzed RnR case. Category: {result.category}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing RnR case: {str(e)}")
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
        Analyze multiple RnR cases in batches.
        
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
                    logger.error(f"Error processing case in batch: {str(e)}")
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
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(cases) + batch_size - 1)//batch_size}")
        
        return results
