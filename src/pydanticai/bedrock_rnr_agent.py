"""
Pydantic AI agent for RnR Reason Code classification using AWS Bedrock.
"""

import json
import logging
from typing import Optional, Dict, Any
import boto3
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.bedrock import BedrockConverseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from .rnr_reason_code_models import RnRReasonCodeAnalysis, RnRAnalysisInput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BedrockRnRAgent:
    """
    Pydantic AI agent for RnR Reason Code classification using AWS Bedrock.
    """
    
    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name: str = "us-west-2",
        max_retries: int = 3,
        **bedrock_kwargs
    ):
        """
        Initialize the Bedrock RnR Agent.
        
        Args:
            model_id: Bedrock model ID to use
            region_name: AWS region name
            max_retries: Maximum number of retries for API calls
            **bedrock_kwargs: Additional arguments for Bedrock client
        """
        self.model_id = model_id
        self.region_name = region_name
        self.max_retries = max_retries
        
        # Initialize Bedrock client
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=region_name,
            **bedrock_kwargs
        )
        
        # Initialize Bedrock model for Pydantic AI
        self.model = BedrockConverseModel(
            model_id=model_id,
            region=region_name,
            **bedrock_kwargs
        )
        
        # Create the Pydantic AI agent
        self.agent = Agent(
            model=self.model,
            result_type=RnRReasonCodeAnalysis,
            system_prompt=self._get_system_prompt()
        )
    
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def analyze_rnr_case(
        self,
        dialogue: str,
        shiptrack: str,
        max_estimated_arrival_date: Optional[str] = None
    ) -> RnRReasonCodeAnalysis:
        """
        Analyze an RnR case and return structured classification.
        
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
            
            # Prepare the prompt
            prompt = f"""Please analyze the following RnR case:

Dialogue: {analysis_input.dialogue}

Ship Track Events: {analysis_input.shiptrack}

Estimated Delivery: {analysis_input.max_estimated_arrival_date or 'Not provided'}

Provide your analysis following the structured format with proper JSON structure for the Pydantic model."""

            # Run the agent
            result = await self.agent.run(prompt)
            
            logger.info(f"Successfully analyzed RnR case. Category: {result.data.category}")
            return result.data
            
        except Exception as e:
            logger.error(f"Error analyzing RnR case: {str(e)}")
            raise

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
            
            # Prepare the prompt
            prompt = f"""Please analyze the following RnR case:

Dialogue: {analysis_input.dialogue}

Ship Track Events: {analysis_input.shiptrack}

Estimated Delivery: {analysis_input.max_estimated_arrival_date or 'Not provided'}

Provide your analysis following the structured format with proper JSON structure for the Pydantic model."""

            # Run the agent synchronously
            result = self.agent.run_sync(prompt)
            
            logger.info(f"Successfully analyzed RnR case. Category: {result.data.category}")
            return result.data
            
        except Exception as e:
            logger.error(f"Error analyzing RnR case: {str(e)}")
            raise

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
