# Day 3 — Structured Outputs

## Mental Model

Text
→ LLM
→ schema-constrained JSON
→ Pydantic validation
→ typed Python object
→ application

## Key Ideas

- Schema = expected data contract.
- Pydantic validates external/untrusted data.
- Literal restricts values to known options.
- Field adds constraints such as ranges and minimum lengths.
- Structured output makes LLM responses usable by software.
- Schema correctness is different from semantic correctness.