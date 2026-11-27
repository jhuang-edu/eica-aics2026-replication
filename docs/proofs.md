# EICA Formal Proofs, Advisory Routing, and Complexity Analysis

> Companion technical appendix to *Ontology-Constrained LLM Evidence Extraction and Finite-State Decision Support System for Compliance-Critical Talent Screening* (IEEE AI-CS 2026).  
> Executable reference implementation: [`src/eica_engine.py`](../src/eica_engine.py) | Unit tests: [`tests/test_eica_unit.py`](../tests/test_eica_unit.py) (65/65 passing).

---

## 1. Pre-Gate Advisory Routing vs. Runtime Hard FSM Gate

Let $c = (\ell_1, \dots, \ell_m) \in \mathcal{C}$ denote a line-indexed de-identified dossier, $R = \{r_1, \dots, r_n\}$ the role requirement ontology partitioned into extractable slots $R_{\mathrm{ext}} = \{r \in R : \tau(r) \in \{\mathrm{M}, \mathrm{P}\}\}$ and masked tacit slots $R_{\mathrm{V}} = \{r \in R : \tau(r) = \mathrm{V}\}$, and $\Sigma = \{\textit{verified}, \textit{not\_provided}, \textit{conflicting}\}$ the closed status enumeration.

After Stage 2 (`LLMExtract` + `EnforceMaskAndStrip`) and Stage 2b (`DeterministicVerifier`), the verified state is $\langle c, \boldsymbol{\sigma}^*, \mathcal{F}(c), \mathcal{A}_{\mathrm{hist}} \rangle$. Under asymmetric compliance loss
$$\mathcal{L}(D, Y) = \lambda_{\mathrm{FA}}\cdot\mathbb{I}(D=1,Y=0) + \lambda_{\mathrm{FR}}\cdot\mathbb{I}(D=0,Y=1) + \lambda_{\mathrm{Esc}}\cdot\mathbb{I}(D=\mathrm{Esc})$$
with design cost parameterization $\lambda_{\mathrm{FA}} \gg \lambda_{\mathrm{FR}} > \lambda_{\mathrm{Esc}}$ (default normalized evaluation $\lambda_{\mathrm{FR}}=1.0$, $\lambda_{\mathrm{Esc}}=0.25$, $\lambda_{\mathrm{FA}} \in \{1, 2, 5, 10, 20\}$), the **pre-gate advisory routing policy** $D_{\mathrm{rec}}(c)$ recommends disposition prior to human action selection:

$$
D_{\mathrm{rec}}(c) = \begin{cases}
\mathrm{Esc}, & \text{if } \mathcal{F}(c) \neq \varnothing \;\vee\; \exists r\big(\sigma^*(c,r) = \textit{conflicting}\big) \\
              & \quad \vee\; \exists r\big(\tau(r)=\mathrm{M} \wedge \sigma^*(c,r)=\textit{not\_provided} \wedge E(c)=1\big), \\
0,            & \text{else if } \exists r\big(\tau(r)=\mathrm{M} \wedge \sigma^*(c,r)=\textit{not\_provided} \wedge E(c)=0\big), \\
\Psi(\boldsymbol{\sigma}^*, \mathbf{x}_{\mathrm{rec}}), & \text{otherwise},
\end{cases}
$$

where $E(c) \in \{0, 1\}$ indicates whether documented cross-standard equivalence claims appear in Tier-$\mathrm{P}$ slots or recruiter escalation notes, and $\Psi(\boldsymbol{\sigma}^*, \mathbf{x}_{\mathrm{rec}}) \to a \in \mathcal{A}$ denotes deliberative human action selection over $\mathcal{A} = \{\textit{Confirm}(D), \textit{Adjust}(D, \boldsymbol{\sigma}', \mathrm{rsn}), \textit{Escalate}\}$ ($D \in \{0, 1\}$).

Runtime enforcement is governed unconditionally by the deterministic finite-state transition function $\delta(s, a)$:
- Mandatory Guard: $\mathcal{G}_{\mathrm{M}}(\boldsymbol{\sigma}) \equiv \forall r\big(\tau(r) = \mathrm{M} \Rightarrow \sigma(c, r) = \textit{verified}\big)$.
- Hard Conflict-Escalation Lock: $\mathcal{G}_{\mathrm{esc}}(\boldsymbol{\sigma}) \equiv \big(\mathcal{F}(c) = \varnothing \;\wedge\; \forall r \in R_{\mathrm{ext}},\, \sigma(c, r) \neq \textit{conflicting}\big)$.

---

## 2. Formal Safety Invariants and Complete Proofs

### Proposition 1 (Zero Tacit Extrapolation and Score Leakage)
> **Statement.** For any raw payload $\mathrm{Card}_{\mathrm{raw}}$, `EnforceMaskAndStrip` followed by `DeterministicVerifier` guarantees zero surviving score/ranking keys and $\sigma^*(c, r) = \varnothing$ for all $r \in R_{\mathrm{V}}$.

**Proof.**
1. **Compile-time schema omission.** `OntologySchemaMasker.compile_generation_schema(role)` emits a strict JSON Schema (`additionalProperties: false`) whose `req_id` enumeration contains only $R_{\mathrm{ext}} = \{r \in R : \tau(r) \in \{\mathrm{M}, \mathrm{P}\}\}$, structurally omitting $R_{\mathrm{V}}$ and omitting all scalar score or ranking fields (`overall_fit_score`, `rank`, `score`, `fit_score`).
2. **Host-side top-level and entry-level projection.** Regardless of whether an unconstrained or adversarial extractor violates the JSON Schema, `OntologySchemaMasker.enforce_mask_and_validate(raw_card, role)` projects the top-level dictionary strictly onto `{"candidate_id", "role", "entries"}`, stripping any top-level scalar keys while recording mask-audit flags $\mathcal{F}_{\mathrm{mask}}$. Each surviving entry dictionary is reconstructed strictly over the 6 whitelisted schema fields (`req_id`, `tier`, `evidence`, `anchor_lines`, `status`, `claim_text`), eliminating any entry-nested score or ranking keys.
3. **Deterministic Tier-$\mathrm{V}$ blanking.** Any injected payload targeting $r \in R_{\mathrm{V}}$ triggers `TACIT_EXTRAPOLATION_BLOCKED` in $\mathcal{F}_{\mathrm{mask}}$ during Stage 1 masking (`OntologySchemaMasker.enforce_mask_and_validate`, `src/eica_engine.py:149`) and is replaced by an empty entry with `evidence = ""`, `anchor_lines = []`, `status = None` ($\sigma = \varnothing$).
4. **Verifier & FSM closure.** In Stage 2b (`DeterministicVerifier.verify_card`, `src/eica_engine.py:1227`; Algorithm 1 line 5 in the 4-page edition, lines 7–8 in the 9-page edition), any non-blank Tier-$\mathrm{V}$ slot (including tampered `claim_text` or `status`) is deterministically reset to $\sigma^*(c, r) = \varnothing$ with `TACIT_EXTRAPOLATION_BLOCKED` (abbreviated `TACIT_BLOCKED` in the 4-page Algorithm 1 line 5) appended to $\mathcal{F}(c)$. In Stage 3 (`FSMDecisionGate.step`), `Adjust` restricts status mutations exclusively to $r \in R_{\mathrm{ext}}$ (`tier in ("M", "P")`). Hence no transition or payload can produce a surviving score/ranking key or a non-blank Tier-$\mathrm{V}$ status. $\blacksquare$

---

### Proposition 2 (Zero Unresolved-Conflict and Unverified-Mandatory Admission)
> **Statement.** Under $\delta(s, a)$ with the hard escalation lock active (`enforce_hard_escalation_lock = True`), terminal admission $s_{\mathrm{term}}(1, \cdot)$ is reachable only if $\mathcal{G}_{\mathrm{M}}(\boldsymbol{\sigma}^*) \wedge \mathcal{G}_{\mathrm{esc}}(\boldsymbol{\sigma}^*)$ holds under `Confirm(1)`, or $\mathcal{G}_{\mathrm{esc}}(\boldsymbol{\sigma}^*) \wedge \mathcal{G}_{\mathrm{M}}(\boldsymbol{\sigma}') \wedge \mathcal{G}_{\mathrm{esc}}(\boldsymbol{\sigma}')$ holds with $|\mathrm{rsn}| \ge L_{\min}$ and $\boldsymbol{\sigma}' \in \Sigma^{|R_{\mathrm{ext}}|}$ under `Adjust(1, \boldsymbol{\sigma}', \mathrm{rsn})`.

**Proof.**
Examine all actions $a \in \mathcal{A}$ plus invalid actions under `FSMDecisionGate.step`:
1. `Escalate` unconditionally transitions to $s_{\mathrm{esc}}$ (non-terminal admission).
2. `Confirm(0)` and `Adjust(0, ...)` either transition to terminal rejection $s_{\mathrm{term}}(0, \cdot)$ or block into $s_{\mathrm{block}}$; neither reaches $s_{\mathrm{term}}(1, \cdot)$.
3. Invalid action names, non-binary `proposed_decision`, invalid status strings outside $\Sigma$, or short justifications ($|\mathrm{rsn}| < L_{\min}$) immediately transition to $s_{\mathrm{block}}(\mathrm{err})$.
4. Under `Confirm(1)`, lines 1371–1385 of `src/eica_engine.py` explicitly check:
   - `if self.enforce_hard_escalation_lock and self._has_unresolved_conflict(card): return s_block("ERR_UNRESOLVED_CONFLICT_LOCK")` — enforcing $\mathcal{G}_{\mathrm{esc}}(\boldsymbol{\sigma}^*)$;
   - `if not self._mandatory_guard_holds(entries): return s_block("ERR_MANDATORY_GUARD_FAILED")` — enforcing $\mathcal{G}_{\mathrm{M}}(\boldsymbol{\sigma}^*)$.
5. Under `Adjust(1, \boldsymbol{\sigma}', \mathrm{rsn})`, lines 1392–1438 enforce in sequence:
   - $|\mathrm{rsn}| \ge L_{\min}$ (`ERR_JUSTIFICATION_TOO_SHORT`);
   - pre-adjustment $\mathcal{G}_{\mathrm{esc}}(\boldsymbol{\sigma}^*)$ (`ERR_UNRESOLVED_CONFLICT_LOCK`), preventing recruiters from silently editing away a verifier conflict or flagged entry to force admission;
   - closed-enum membership $\boldsymbol{\sigma}' \in \Sigma^{|R_{\mathrm{ext}}|}$ (`ERR_INVALID_ENUM_STATUS`);
   - post-adjustment $\mathcal{G}_{\mathrm{esc}}(\boldsymbol{\sigma}')$ and $\mathcal{G}_{\mathrm{M}}(\boldsymbol{\sigma}')$.
Therefore, every path reaching $s_{\mathrm{term}}(1, \cdot)$ satisfies the stated invariant; any violation deterministically blocks into $s_{\mathrm{block}}(\mathrm{err})$, compelling `Escalate` $\to s_{\mathrm{esc}}$. $\blacksquare$

---

## 3. Time and Space Complexity Derivation

- **Offline Regulatory Indexing (`HybridRegulatoryRAG.__init__`):** Tokenizes each active clause $d \in \mathcal{D}_{\mathrm{pin}}$ into lexical tokens and character 3-grams, computing inverse document frequency (IDF) weights and normalized sparse TF-IDF vectors in $O(|\mathcal{D}_{\mathrm{pin}}|\,\bar{w})$ time and space, where $\bar{w}$ is mean clause token/3-gram length ($|\mathcal{D}_{\mathrm{pin}}| = 12$ active clauses).
- **Online Stage 1 — Host Schema Masking (`enforce_mask_and_validate`):** Single pass over raw entries plus ontology slot reconciliation runs in $O(|\mathrm{Card}_{\mathrm{raw}}| + |R|)$ time.
- **Online Stage 1 — Hybrid Sparse Retrieval (`retrieve`):** Tokenizes query $q$ in $O(|q|)$, scores all $d \in \mathcal{D}_{\mathrm{pin}}$ via normalized BM25 (per-query TF count over $\bar{w}$ tokens, reducible by $\bar{w}$ via offline TF precomputation) and sparse cosine dot product, and sorts scores in $O(|q| + |\mathcal{D}_{\mathrm{pin}}|(\bar{w} + |q| + \log |\mathcal{D}_{\mathrm{pin}}|))$.
- **Online Stage 2b — Deterministic Verifier (`DeterministicVerifier.verify_card`):** For each extractable slot $r \in R_{\mathrm{ext}}$, slices at most $i_e - i_s + 1 \le 3$ bounded lines $t_{\mathrm{src}}$, runs contiguous substring/token-anchored containment and sliding-window `SequenceMatcher` similarity ($O(|e|\,|t_{\mathrm{src}}|^2)$ worst case over bounded short strings), plus $P_{\mathrm{lex}}$ compiled lexical/regex verb, negation, and credential patterns ($O(P_{\mathrm{lex}}\,|t_{\mathrm{src}}|)$), yielding $O(|R_{\mathrm{ext}}|\,(|e|\,|t_{\mathrm{src}}|^2 + P_{\mathrm{lex}}\,|t_{\mathrm{src}}|))$.
- **Online Stage 3 — FSM Gate (`FSMDecisionGate.step`):** Linear scan over $|R|$ entries for $\mathcal{G}_{\mathrm{M}}$ and $\mathcal{G}_{\mathrm{esc}}$, running in $O(|R|)$ time and $O(|R|)$ memory.
- **GPU VRAM Overhead:** Stages 1, 2b, and 3 execute strictly on host CPU, adding **0 GiB GPU VRAM** alongside the $21.5 / 24.0$~GiB single-RTX-3090 extraction envelope ($19.3$~GiB 4-bit AWQ weights $+ 2.0$~GiB KV cache $+ 0.2$~GiB runtime overhead).
