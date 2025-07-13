# Developer Guide: Adding a New Step to the Pipeline

**Version**: 2.0  
**Date**: July 12, 2025  
**Author**: MODS Development Team

## Overview

This guide provides a standardized procedure for adding a new step to the pipeline system. Following these guidelines ensures that your implementation maintains consistency with the existing code structure and adheres to our design principles.

Our pipeline architecture follows a specification-driven approach with a four-layer design:

1. **Step Specifications**: Define inputs and outputs with logical names
2. **Script Contracts**: Define container paths for script inputs/outputs
3. **Step Builders**: Connect specifications and contracts via SageMaker
4. **Processing Scripts**: Implement the actual business logic

## Table of Contents

1. [Prerequisites](prerequisites.md)
2. [Step Creation Process](creation_process.md)
3. [Detailed Component Guide](component_guide.md)
   - [Script Contract Development](script_contract.md)
   - [Step Specification Development](step_specification.md)
   - [Step Builder Implementation](step_builder.md)
4. [Design Principles](design_principles.md)
5. [Best Practices](best_practices.md)
6. [Common Pitfalls to Avoid](common_pitfalls.md)
7. [Example](example.md)
8. [Validation Checklist](validation_checklist.md)

## Quick Start

To add a new step to the pipeline:

1. Review the [prerequisites](prerequisites.md) to ensure you have all required information
2. Follow the [step creation process](creation_process.md) to implement all required components
3. Validate your implementation using the [validation checklist](validation_checklist.md)

For detailed guidance on specific components, refer to the relevant sections in the [detailed component guide](component_guide.md).

## Additional Resources

- [Specification-Driven Architecture](../pipeline_design/specification_driven_design.md)
- [Hybrid Design](../pipeline_design/hybrid_design.md)
- [Script-Specification Alignment](../project_planning/script_specification_alignment_prevention_plan.md)
- [Script Contract](../pipeline_design/script_contract.md)
- [Step Specification](../pipeline_design/step_specification.md)
