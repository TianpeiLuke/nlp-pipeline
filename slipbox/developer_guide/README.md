# Pipeline Step Developer Guide

This directory contains comprehensive documentation for developing new steps in our SageMaker-based ML pipeline architecture. The guide is designed to help both new and experienced developers create pipeline steps that align with our architectural patterns and best practices.

## Guide Structure

The developer guide is organized into several interconnected documents:

### Main Documentation

- **[Adding a New Pipeline Step](adding_new_pipeline_step.md)** - The main entry point providing an overview of the step development process

### Process Documentation

- **[Prerequisites](prerequisites.md)** - What you need before starting development
- **[Creation Process](creation_process.md)** - Step-by-step process for adding a new pipeline step

### Component Documentation

- **[Component Guide](component_guide.md)** - Overview of the key components and their relationships
- **[Script Contract Development](script_contract.md)** - Detailed guide for developing script contracts
- **[Step Specification Development](step_specification.md)** - Detailed guide for developing step specifications
- **[Step Builder Implementation](step_builder.md)** - Detailed guide for implementing step builders

### Best Practices and Guidelines

- **[Design Principles](design_principles.md)** - Core design principles to follow
- **[Best Practices](best_practices.md)** - Recommended best practices for development
- **[Common Pitfalls](common_pitfalls.md)** - Common mistakes to avoid
- **[Validation Checklist](validation_checklist.md)** - Comprehensive checklist for validating implementations

### Examples

- **[Example Implementation](example.md)** - Complete example of adding a new pipeline step

## Recommended Reading Order

For new developers, we recommend the following reading order:

1. Start with **[Adding a New Pipeline Step](adding_new_pipeline_step.md)** for an overview
2. Check **[Prerequisites](prerequisites.md)** to ensure you have everything needed
3. Review the **[Creation Process](creation_process.md)** for the step-by-step procedure
4. Read the **[Component Guide](component_guide.md)** to understand component relationships
5. Dive deeper into specific component documentation:
   - **[Script Contract Development](script_contract.md)**
   - **[Step Specification Development](step_specification.md)**
   - **[Step Builder Implementation](step_builder.md)**
6. Study the **[Example Implementation](example.md)** to see how everything fits together
7. Review best practices and guidelines:
   - **[Design Principles](design_principles.md)**
   - **[Best Practices](best_practices.md)**
   - **[Common Pitfalls](common_pitfalls.md)**
8. Use the **[Validation Checklist](validation_checklist.md)** to verify your implementation

## Key Architectural Concepts

Our pipeline architecture follows a specification-driven approach with a four-layer design:

1. **Step Specifications**: Define inputs and outputs with logical names
2. **Script Contracts**: Define container paths for script inputs/outputs
3. **Step Builders**: Connect specifications and contracts via SageMaker
4. **Processing Scripts**: Implement the actual business logic

Understanding these layers and their relationships is crucial for successful step development.

## Using AI to Assist Development

For guidance on using Claude v3 to assist with pipeline step development, see the AI prompt templates in the [../developer_prompts](../developer_prompts) directory.

## Getting Help

If you encounter issues or have questions while developing a new pipeline step:

1. Consult the **[Common Pitfalls](common_pitfalls.md)** document
2. Use the **[Validation Checklist](validation_checklist.md)** to identify potential issues
3. Review the **[Example Implementation](example.md)** for reference
4. Reach out to the architecture team for assistance

## Contributing to the Guide

If you identify gaps in the documentation or have suggestions for improvements:

1. Document your proposed changes
2. Discuss with the architecture team
3. Update the relevant documentation
4. Ensure consistency across all documents

The developer guide is a living document that evolves with our architecture.
