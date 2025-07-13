# Pipeline Step Development with AI Prompts

This directory contains a set of AI prompt templates designed to assist with developing new pipeline steps using Claude v3. These prompts follow a structured workflow with three distinct roles: Planner, Programmer, and Validator.

## Overview of the Workflow

The workflow is divided into three main phases, each with its own prompt template and role:

1. **Planning Phase (Planner)**: Design the new step and create a detailed implementation plan
2. **Implementation Phase (Programmer)**: Write the code for all components based on the plan
3. **Validation Phase (Validator)**: Verify the implementation against design principles and best practices

## How to Use These Prompts

### 1. Planning Phase

Use the `planner_prompt.md` template with Claude to generate a detailed implementation plan:

1. Copy the content of `planner_prompt.md`
2. Replace the placeholder `[INJECT STEP REQUIREMENTS HERE]` with your specific requirements
3. Replace other `[INJECT X DOCUMENT HERE]` placeholders with relevant documentation from our developer guide
4. Submit the prompt to Claude
5. Review and refine the implementation plan

The Planner will analyze requirements and produce a comprehensive plan covering all necessary components and files to update.

### 2. Implementation Phase

Use the `programmer_prompt.md` template with Claude to generate the code:

1. Copy the content of `programmer_prompt.md`
2. Replace `[INJECT PLANNER OUTPUT HERE]` with the plan from the previous phase
3. Replace other `[INJECT X DOCUMENT HERE]` placeholders with relevant documentation
4. Submit the prompt to Claude
5. Review the generated code for correctness

The Programmer will implement all components required by the plan, following our architectural patterns and best practices.

### 3. Validation Phase

Use the `validator_prompt.md` template with Claude to validate the implementation:

1. Copy the content of `validator_prompt.md`
2. Replace `[INJECT PLANNER OUTPUT HERE]` with the original plan
3. Replace `[INJECT PROGRAMMER OUTPUT HERE]` with the implementation code
4. Replace other `[INJECT X DOCUMENT HERE]` placeholders with relevant documentation
5. Submit the prompt to Claude
6. Address any issues identified in the validation report

The Validator will thoroughly review the implementation against our design principles, common pitfalls, and validation checklist.

## Recommended Documentation to Include

For each role, include the most relevant documentation:

### Planner
- `creation_process.md`: Step-by-step process for adding a new step
- `prerequisites.md`: What's needed before starting development
- Example implementations of similar steps

### Programmer
- `component_guide.md`: Overview of component relationships
- `script_contract.md`: Guidelines for script contract development
- `step_specification.md`: Guidelines for specification development
- `step_builder.md`: Guidelines for builder implementation
- `best_practices.md`: Coding best practices
- `example.md`: Complete example implementation

### Validator
- `design_principles.md`: Core design principles to follow
- `common_pitfalls.md`: Common mistakes to avoid
- `validation_checklist.md`: Comprehensive validation checklist

## Placeholder Injection Guide

When preparing the prompts, replace these placeholders with the actual content:

- `[INJECT STEP REQUIREMENTS HERE]`: Your specific requirements for the new step
- `[INJECT CREATION_PROCESS DOCUMENT HERE]`: Content from `slipbox/v2/developer_guide/creation_process.md`
- `[INJECT PREREQUISITES DOCUMENT HERE]`: Content from `slipbox/v2/developer_guide/prerequisites.md`
- `[INJECT COMPONENT_GUIDE DOCUMENT HERE]`: Content from `slipbox/v2/developer_guide/component_guide.md`
- `[INJECT SCRIPT_CONTRACT DOCUMENT HERE]`: Content from `slipbox/v2/developer_guide/script_contract.md`
- `[INJECT STEP_SPECIFICATION DOCUMENT HERE]`: Content from `slipbox/v2/developer_guide/step_specification.md`
- `[INJECT STEP_BUILDER DOCUMENT HERE]`: Content from `slipbox/v2/developer_guide/step_builder.md`
- `[INJECT BEST_PRACTICES DOCUMENT HERE]`: Content from `slipbox/v2/developer_guide/best_practices.md`
- `[INJECT DESIGN_PRINCIPLES DOCUMENT HERE]`: Content from `slipbox/v2/developer_guide/design_principles.md`
- `[INJECT COMMON_PITFALLS DOCUMENT HERE]`: Content from `slipbox/v2/developer_guide/common_pitfalls.md`
- `[INJECT VALIDATION_CHECKLIST DOCUMENT HERE]`: Content from `slipbox/v2/developer_guide/validation_checklist.md`
- `[INJECT EXAMPLE DOCUMENT HERE]`: Content from `slipbox/v2/developer_guide/example.md`
- `[INJECT RELEVANT EXAMPLES HERE]`: Relevant example implementations from codebase
- `[INJECT PLANNER OUTPUT HERE]`: Output from the Planning phase
- `[INJECT PROGRAMMER OUTPUT HERE]`: Output from the Implementation phase

## Best Practices for Working with These Prompts

1. **Progressive Disclosure**: Don't overwhelm Claude with too much context at once. Provide the most relevant documentation for each role.
2. **Chunking**: If documentation is too large, break it into logical chunks and provide the most relevant sections.
3. **Iteration**: Iterate between phases as needed. If validation reveals issues, return to implementation.
4. **Human Review**: Always review AI-generated plans and code before implementing them.
5. **Examples**: Include relevant examples from your codebase to help Claude understand your specific patterns.
6. **Focused Requirements**: Make your step requirements as clear and specific as possible.
7. **Token Management**: Be mindful of Claude's context window limits and prioritize the most important content.

## Expected Outputs

- **Planner**: A structured implementation plan with all components to create/update
- **Programmer**: Complete code for all required components and file updates
- **Validator**: A comprehensive validation report with issues and recommendations

By following this workflow, you can leverage Claude's capabilities to assist with developing new pipeline steps that align with your architecture and best practices.
