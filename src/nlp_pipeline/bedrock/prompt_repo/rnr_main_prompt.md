---
tags:
  - code
  - code_snippet
  - prompt
  - large_language_models
  - python/aws_cdk
keywords: 
topics: 
language: python
date of note: 2025-03-27
---

# RnR Reason Code Classification Prompt

## Overview

This prompt template is designed for PydanticAI-based classification of Return and Refund (RnR) reason codes. It analyzes buyer-seller conversations, shipping events, and delivery timelines to categorize interactions into predefined categories for buyer abuse detection.

## Categories (12 Total)

### 1. TrueDNR (Delivered Not Received)
- Package marked as delivered but buyer claims non-receipt
- Tracking shows delivery but buyer disputes receiving it
- Discussion about delivery location, missing package investigation

### 2. Confirmed_Delay
- Shipment delayed due to uncontrollable external factors
- Delay confirmed by seller or shiptrack status
- Customs, COVID, weather, strikes, accidents, war

### 3. Delivery_Attempt_Failed
- Delivery attempt unsuccessful, package returned to seller
- Confirmed by seller or shiptrack status
- Address issues, recipient unavailable, access restrictions

### 4. Seller_Unable_To_Ship
- Seller offers refund directly due to shipping issues
- Stock unavailable, shipping restrictions, processing problems
- Seller-initiated refund, not buyer cancellation request

### 5. PDA_Undeliverable
- Item stuck in transit without status updates or seller confirmation
- Buyer claims non-receipt while package shows shipped/in-transit
- Potential abuse pattern for early refund manipulation

### 6. PDA_Early_Refund
- Refund given before delivery date
- Product tracking later shows successful delivery
- No product return recorded

### 7. Buyer_Received_WrongORDefective_Item
- Product quality/condition issues requiring actual return
- Damaged, defective, wrong item, missing parts
- Eventually becomes actual return process

### 8. Returnless_Refund
- Refund given without requiring customer to return product
- Liquids, gels, hazardous materials, cheap items
- Seller explicitly allows item retention

### 9. BuyerCancellation
- Buyer cancels order before delivery for their own reasons
- Change of plan, late delivery concerns, payment issues
- No returns involved as buyer hasn't received item yet

### 10. Return_NoLongerNeeded
- Post-delivery return of good quality items no longer needed
- Changed mind after receiving, size issues, duplicate purchase
- Eventually becomes actual return process

### 11. Product_Information_Support
- General product information and support requests
- Invoices, pricing inquiries, usage instructions, troubleshooting
- Not related to refund or return events

### 12. Insufficient_Information
- Ultimate "I don't know" category when context is missing
- Incomplete dialogue, missing ship track, no seller engagement
- Available data insufficient for other categories

## Prompt Template

```text
You are an expert in analyzing buyer-seller messaging conversations and shipping logistics. Your task is to classify the interaction based on message content, shipping events, and delivery timing into one of several predefined categories.

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

Analysis Instructions:

Please analyze:
Dialogue: {dialogue}
Ship Track Events: {shiptrack}
Estimated Delivery: {max_estimated_arrival_date}

**CRITICAL FORMAT REQUIREMENT: You MUST follow the exact output format specified below. Any deviation from this format will cause parsing errors and system failures.**

**MANDATORY OUTPUT STRUCTURE - DO NOT DEVIATE:**

1. Category: [Exactly one of: TrueDNR, Confirmed_Delay, Delivery_Attempt_Failed, Seller_Unable_To_Ship, PDA_Undeliverable, PDA_Early_Refund, Buyer_Received_WrongORDefective_Item, Returnless_Refund, BuyerCancellation, Return_NoLongerNeeded, Product_Information_Support, Insufficient_Information]

2. Confidence Score: [Number between 0.00 and 1.00]

3. Key Evidence:
   * Message Evidence:
     [sep] [Quote or description of relevant message content]
     [sep] [Additional message evidence if available]
   * Shipping Evidence:
     [sep] [Relevant tracking events with timestamps]
     [sep] [Additional shipping evidence if available]
   * Timeline Evidence:
     [sep] [Chronological analysis of events]
     [sep] [Timing comparisons]

4. Reasoning:
   * Primary Factors:
     [sep] [Main reasons for classification]
     [sep] [Key decision points]
   * Supporting Evidence:
     [sep] [Additional evidence supporting the classification]
     [sep] [Corroborating details]
   * Contradicting Evidence:
     [sep] [Any evidence that conflicts with the classification]
     [sep] [Write "None" if no contradicting evidence exists]

**STRICT FORMATTING RULES - FAILURE TO FOLLOW WILL CAUSE SYSTEM ERROR:**
- Use EXACT section headers with numbers and colons (1. Category:, 2. Confidence Score:, etc.)
- Category name must match EXACTLY from the provided list (case-sensitive)
- Confidence score MUST be decimal format (e.g., 0.85, NOT 85% or "high")
- Each evidence/reasoning item MUST start with "[sep] " (including the space)
- Use asterisk (*) for subsection headers with EXACT spacing shown above
- NO semicolons (;) anywhere in the response
- NO additional formatting, markdown, or extra text outside the required structure
- Multiple evidence items MUST be separated by individual [sep] tokens on separate lines

## Required Output Format

**CRITICAL: Follow this exact format for automated parsing**

```
1. Category: [EXACT_CATEGORY_NAME]

2. Confidence Score: [0.XX]

3. Key Evidence:
   * Message Evidence:
     [sep] [Evidence item 1]
     [sep] [Evidence item 2]
   * Shipping Evidence:
     [sep] [Evidence item 1]
     [sep] [Evidence item 2]
   * Timeline Evidence:
     [sep] [Evidence item 1]
     [sep] [Evidence item 2]

4. Reasoning:
   * Primary Factors:
     [sep] [Factor 1]
     [sep] [Factor 2]
   * Supporting Evidence:
     [sep] [Supporting item 1]
     [sep] [Supporting item 2]
   * Contradicting Evidence:
     [sep] [Contradicting item 1 OR "None"]
```

**Formatting Rules:**
- Use exact section headers with numbers and colons
- Category name must match exactly from provided list
- Confidence score as decimal (e.g., 0.85, not 85% or "high")
- Each evidence/reasoning item starts with "[sep] " (including space)
- Use asterisk (*) for subsection headers with exact spacing
- No semicolons (;) anywhere in response
- No additional formatting or markdown

**Example Output:**

```
1. Category: TrueDNR

2. Confidence Score: 0.92

3. Key Evidence:
   * Message Evidence:
     [sep] [BUYER]: Hello, I have not received my package, but I see the order shows that it has been delivered, why?
     [sep] [BUYER]: But I did not find any package, please refund me, thank you
   * Shipping Evidence:
     [sep] [Event Time]: 2025-02-21T17:40:49.323Z [Ship Track Event]: Delivered to customer
     [sep] No further shipping events after delivery confirmation
   * Timeline Evidence:
     [sep] Delivery confirmation on 2025-02-21 17:40
     [sep] Buyer reports non-receipt starting 2025-02-25 07:14

4. Reasoning:
   * Primary Factors:
     [sep] Tracking shows package was delivered successfully
     [sep] Buyer explicitly states they did not receive the package after delivery scan
   * Supporting Evidence:
     [sep] Buyer requests refund due to missing package
     [sep] No evidence of buyer receiving wrong/defective item
   * Contradicting Evidence:
     [sep] None
```

## Classification Guidelines

### 1. Output Format Requirements

**Category Selection:**
- Choose exactly ONE category from the provided list
- Category name must match exactly (case-sensitive)

**Confidence Score:**
- Provide as decimal number between 0.00 and 1.00 (e.g., 0.95)
- Base confidence for complete data: 0.7-1.0
- Missing one field: reduce by 0.1-0.2
- Missing two fields: reduce by 0.2-0.3
- Minimum confidence threshold: 0.5

**Key Evidence Format:**
- Use exactly three subsections: Message Evidence, Shipping Evidence, Timeline Evidence
- Start each evidence item with "[sep]" token
- Include specific quotes and timestamps where available
- Do NOT use semicolons (;) in the response
- Multiple pieces of evidence separated by [sep] token

**Reasoning Format:**
- Use exactly three subsections: Primary Factors, Supporting Evidence, Contradicting Evidence
- Start each reasoning item with "[sep]" token
- Write "None" if no contradicting evidence exists

### 2. Shiptrack Parsing Rules

**Multiple Shipment Structure:**
- Multiple shipment sequences separated by shipment IDs
- Each sequence starts with "[bom] [Shipment ID]:* [eom]"
- Each sequence ends with "[bom] End of Ship Track Event for* [eom]"
- Contains chronologically ordered events between markers

**Analysis Approach:**
- Process each shipment sequence separately
- Compare delivery events (EVENT_301) across all sequences
- Use the latest delivery timestamp when multiple sequences exist
- Consider all tracking events when evaluating shipping status
- Pay attention to common tracking IDs across sequences
- Look for discrepancies between sequences with same tracking ID
- Use supplement codes for additional context about event locations

**Key Event Codes:**
- EVENT_101: Initial carrier receipt/pickup
- EVENT_102: Carrier pickup from fulfillment center
- EVENT_201: Shipment arrived at carrier facility
- EVENT_202: Shipment departed from carrier facility
- EVENT_301: Delivered to customer
- EVENT_302: Shipment is out for delivery
- EVENT_304: Delivery attempted (addressee not at home)
- EVENT_308: Shipment available for pickup
- EVENT_390: Package delivered and cash received (COD)
- EVENT_401: Driver failed attempt due to address problem
- EVENT_402: Shipment delayed in customs clearance process
- EVENT_404: Shipment delayed due to carrier network issues
- EVENT_406: Held by carrier
- EVENT_407: Consignee refused to accept package
- EVENT_409: Carrier lost the shipment
- EVENT_414: Package mis-sorted, delivery may be delayed
- EVENT_416: Parcel undeliverable and may be disposed of
- EVENT_420: Delay due to weather or natural disaster
- EVENT_423: Package damaged, will not complete delivery
- EVENT_434: Amazon requested return to sender

### 3. Missing Data Handling

**When Dialogue is Empty but Shiptrack Exists:**
- Focus on shipping events and timeline
- Categories determinable from ship track alone:
  * **TrueDNR**: EVENT_301 (delivered) confirms delivery occurred
  * **Confirmed_Delay**: EVENT_402 (customs delay), EVENT_420 (weather delay), EVENT_404 (carrier network delay)
  * **Delivery_Attempt_Failed**: EVENT_304 (addressee not home), EVENT_401 (address problem), EVENT_416 (undeliverable)
  * **PDA_Early_Refund**: Can verify delivery timestamp vs estimated delivery date
- Categories requiring dialogue context:
  * **PDA_Undeliverable**: Need buyer claims of non-receipt
  * **Seller_Unable_To_Ship**: Need seller communication
  * **Buyer_Received_WrongORDefective_Item**: Need buyer quality complaints
  * **Returnless_Refund**: Need explicit seller permission to keep item
  * **BuyerCancellation**: Need buyer cancellation request
  * **Return_NoLongerNeeded**: Need buyer return request
  * **Product_Information_Support**: Need information requests
- Reduce confidence score by 0.1-0.2 for ship track only classifications
- Cannot verify buyer reactions, claims, or seller responses

**When Shiptrack is Empty but Dialogue Exists:**
- Focus on message content and reported issues
- Categories determinable from dialogue alone:
  * **Buyer_Received_WrongORDefective_Item**: Buyer reports quality/condition issues, damaged/defective items, wrong items received
  * **Returnless_Refund**: Seller explicitly offers refund without return ("keep the item", "no need to return")
  * **BuyerCancellation**: Buyer requests order cancellation before delivery
  * **Return_NoLongerNeeded**: Buyer requests return of received items (no longer needed, size issues, changed mind)
  * **Product_Information_Support**: Information requests (invoices, specifications, usage instructions, troubleshooting)
  * **Seller_Unable_To_Ship**: Seller proactively contacts buyer about inability to ship (out of stock, shipping restrictions)
- Categories requiring shiptrack confirmation:
  * **TrueDNR**: Need EVENT_301 to confirm delivery occurred
  * **Confirmed_Delay**: Need shiptrack delay events (EVENT_402, EVENT_420, EVENT_404)
  * **Delivery_Attempt_Failed**: Need failed delivery events (EVENT_304, EVENT_401, EVENT_416)
  * **PDA_Undeliverable**: Need shiptrack showing shipped/in-transit status without delivery
  * **PDA_Early_Refund**: Need delivery timestamp comparison with refund timing
- Use **Insufficient_Information** when:
  * Dialogue is corrupted, unreadable, or non-language content
  * Messages are cut off or incomplete
  * No clear buyer/seller engagement visible
  * Available dialogue insufficient for classification
- Reduce confidence score by 0.1-0.2 for dialogue-only classifications
- Cannot verify actual delivery status, shipping events, or precise timing

**When Estimated Delivery Date is Empty:**
- Cannot make timing-based classifications
- Avoid categories requiring EDD comparison
- Reduce confidence score by 0.1

### 4. Category Priority Hierarchy

**Classification Decision Tree - Apply in Order:**

**Step 1: Determine Transaction Type**
- Does the case involve refund or return? → Continue to Step 2
- No financial transaction (information only)? → **Product_Information_Support**
- Cannot determine from available data? → **Insufficient_Information**

**Step 2: Determine Return Requirement**
- Refund granted WITHOUT return expected? → **Group A: Refund-Only Cases**
- Refund granted WITH return required? → **Group B: Return + Refund Cases**

**Group A: Refund-Only Cases (Highest Abuse Risk)**
*Priority Order within Group A:*
1. **PDA_Early_Refund** - Refund before delivery (verify timeline evidence)
2. **PDA_Undeliverable** - Item lost/stuck, no delivery confirmation
3. **TrueDNR** - Delivered but buyer disputes receipt
4. **Confirmed_Delay** - External delay factors confirmed
5. **Delivery_Attempt_Failed** - Failed delivery, returned to seller
6. **Seller_Unable_To_Ship** - Seller cannot fulfill order
7. **BuyerCancellation** - Pre-delivery buyer cancellation

**Group B: Return + Refund Cases (Medium Risk)**
*Priority Order within Group B:*
1. **Returnless_Refund** - Exception case: refund without return (verify explicit permission)
2. **Buyer_Received_WrongORDefective_Item** - Quality issues requiring return
3. **Return_NoLongerNeeded** - Post-delivery unwanted item return

**Group C: Information/Support Only (Lowest Risk)**
1. **Product_Information_Support** - Information requests, no financial impact
2. **Insufficient_Information** - Missing context for classification

**Key Decision Criteria:**
- **Timeline Evidence**: Required for PDA_Early_Refund (refund timestamp < delivery timestamp)
- **Return Record**: Must verify absence for Group A categories
- **Delivery Confirmation**: Critical for TrueDNR vs PDA_Undeliverable distinction
- **Explicit Permission**: Required for Returnless_Refund (seller states "keep the item")
- **Quality Claims**: Distinguishes defective items from unwanted items

### 5. Evidence Requirements

**Message Evidence Must Include:**
- Direct quotes from dialogue with speaker identification
- Timestamps for all messages when available
- Key claims, disputes, or requests from buyer/seller
- Specific language indicating category criteria

**Shipping Evidence Must Include:**
- All tracking events listed chronologically
- Delivery status and attempts with timestamps
- Estimated delivery date when available
- Return tracking events if applicable
- Event codes and locations

**Timeline Evidence Must Show:**
- Clear chronological sequence of events
- Order placement → Shipping → Delivery attempts → Refund timing
- Message exchanges relative to shipping events
- Comparison of key timestamps (refund vs delivery, etc.)

Classification Rules:
- Choose exactly ONE category
- Provide confidence score as a decimal number (e.g., 0.95)
- List evidence as bullet points starting with "[sep]"
- Include specific quotes and timestamps where available
- Separate evidence types clearly under appropriate headers
- Ensure all sections are properly formatted for automated parsing
- Do NOT use semicolons (;) in the response
- Multiple pieces of evidence should be separated by [sep] token
- If Ship Track Events and Estimated Delivery are missing or empty, make classification decision based on Dialogue content alone and adjust confidence score accordingly
**Follow the Required Output Format above exactly as shown.**
```

## Integration with PydanticAI

This prompt is designed to work with the PydanticAI framework using the following models:

- `ReasonCodeCategory`: Enum defining all 12 categories
- `RnRReasonCodeAnalysis`: Main output model with structured evidence and reasoning
- `EvidenceSection`: Structured evidence with automatic [sep] formatting
- `ReasoningSection`: Structured reasoning with validation
- `RnRAnalysisInput`: Input validation for dialogue, shiptrack, and delivery date

The prompt ensures compatibility with automated parsing and validation through PydanticAI's structured output capabilities.

## Related Files

- `src/buyer_abuse_nlp/pydanticai/rnr_reason_code_models.py`: Pydantic models
- `src/buyer_abuse_nlp/pydanticai/bedrock_rnr_agent.py`: PydanticAI agent implementation
- `src/buyer_abuse_nlp/pydanticai/rnr_bedrock_main.py`: Main processor with inference profile support
