# EICA Constrained Extraction — Reference Prompt Template (`LLMExtract`)

> **Provenance & Disclosure Notice:**  
> The historical Chinese production prompt deployed inside the partner medical device manufacturer (Jan. 2025–Jun. 2026) incorporates proprietary internal SOP phrasing and is withheld under enterprise confidentiality and China PIPL governance controls. This file provides an **open English Reference Prompt Template** faithfully reproducing the exact structural, ontological, RAG-binding, and schema-masking constraints of **Algorithm 1 (`LLMExtract`, Section II in 4-page / Section III in 9-page edition)** for open-science inspection and local vLLM reproduction ([`vllm_config.yaml`](../vllm_config.yaml); `code/vllm_config.yaml` in the full workspace).

---

## 1. System Prompt (Reference Template)

```text
You are a constrained compliance evidence extractor for medical device talent screening under ISO 13485:2016 (China GB/T 42061-2022), NMPA GMP, and China State Council Order No. 739/797.

STRICT OPERATIONAL RULES:
1. Extract ONLY verbatim resume quotes and exact 1-based line anchors for the extractable Tier-M (Mandatory) and Tier-P (Preferred Equivalent) requirement IDs listed in the JSON Schema.
2. NEVER output scalar fit scores, candidate rankings, or hire/reject verdicts.
3. NEVER infer, fabricate, or emit submerged tacit traits (Tier-V interview-only competencies). Tier-V slots are structurally excluded from the output schema.
4. Use ONLY the closed three-state status enum:
   - "verified": verbatim resume text directly satisfies the requirement action-verb level and statutory credential proof (when required).
   - "not_provided": the dossier contains no relevant evidence for the requirement.
   - "conflicting": the dossier contains weak/ambiguous verbs, negation, training-only phrasing without statutory credential proof, or cross-standard equivalence requiring manager review.
5. Do NOT inflate weak participation verbs ("participated", "assisted", "trained") into certified or leadership claims ("led", "independent", "certified").
6. Ground any cross-standard equivalence (e.g., ISO 9001 or pharmaceutical sterile GMP vs. ISO 13485) strictly against the Top-3 version-pinned regulatory clauses provided in the prompt. Do NOT cite superseded regulations (e.g., Order No. 650).
7. Return strictly valid JSON conforming to the provided JSON Schema with additionalProperties=false.
```

---

## 2. User Prompt Structure (Programmatically Assembled by `build_extraction_prompt`)

```text
Target Role: {role}
Candidate ID: {candidate_id}

### Version-Pinned Regulatory Clauses (Top-3 Active D_pin)
{pinned_rag_clauses}

### Extractable Competency Requirements (Tier M and Tier P Only; Tier V Masked)
{extractable_requirements}

### Numbered De-Identified Candidate Dossier
{numbered_dossier_lines}

### Output Requirement
Return a single JSON object matching the compiled EICA JSON Schema (exact keys: candidate_id, role, entries).
```

---

## 3. Decoding & Serving Binding (`../vllm_config.yaml`)

- **Model:** `Qwen/Qwen2.5-32B-Instruct` (4-bit AWQ; exact Jan. 2025 enterprise local checkpoint SHA was not archived in production logs)
- **Serving Parameters:** `temperature=0.0`, `seed=42`, `max_tokens=2048`, `max_model_len=8192`, `batch_size=1`, `retry <= 2`
- **Structured Decoding:** `response_format={"type": "json_schema", "json_schema": {"name": "eica_evidence_card", "strict": True, "schema": <OntologySchemaMasker.compile_generation_schema(role)>}}`
- **Host-Side Post-Assembly Guard:** Every parsed payload is passed through `OntologySchemaMasker.enforce_mask_and_validate()` followed by `DeterministicVerifier.verify_card()` (Stages 1, 2b, and 3 execute deterministically on host CPU).
