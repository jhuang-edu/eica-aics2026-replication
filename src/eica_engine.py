# -*- coding: utf-8 -*-
"""
EICA Core System Architecture Implementation (IEEE AI-CS 2026)
=============================================================
Executable implementation of the four algorithmic components of EICA:
  1. Three-Tier Competency Ontology & JSON-Schema Hard Masking (M / P / V)
  2. Version-Pinned Hybrid Regulatory RAG (BM25 + Subword-Vector Cosine)
  3. Deterministic Span-Anchor, Verb-Escalation & Credential Conflict Verifier
  4. Finite-State Human-in-the-Loop Decision Gate (FSM) with Hard Escalation Lock

Authors: Junjun Huang*, Ning Chen
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple


# =============================================================================
# 1. Three-Tier Competency Ontology & Schema Masker
# =============================================================================

CLOSED_STATUS_ENUM = ("verified", "not_provided", "conflicting")


@dataclass(frozen=True)
class CompetencyReq:
    req_id: str
    label: str
    tier: str  # "M" (Mandatory), "P" (Preferred Equivalent), "V" (Tacit / Interview)
    min_verb_tier: int = 2  # 1=weak, 2=moderate, 3=strong/certified
    requires_cert_proof: bool = False
    query_terms: Tuple[str, ...] = ()


class CompetencyOntology:
    """Three-tier competency ontology grounded in ISO 13485:2016 and Order 739/797."""

    ONTOLOGY: Dict[str, List[CompetencyReq]] = {
        "QA/QC": [
            CompetencyReq("QA_DEG", "Bachelor degree in biomedical/materials/engineering", "M", 1, False, ("degree", "qualification", "ISO 13485 6.2")),
            CompetencyReq("QA_EXP", "3+ years Class II/III medical device QA experience", "M", 2, False, ("Class II", "Class III", "QA", "Order 739")),
            CompetencyReq("QA_CERT", "ISO 13485 internal auditor certification", "M", 3, True, ("ISO 13485", "internal auditor", "certificate", "clause 8.2.4")),
            CompetencyReq("QA_VAL", "Process validation (IQ/OQ/PQ) and sterile/cleanroom control", "P", 2, False, ("IQ/OQ/PQ", "process validation", "sterile", "ISO 13485 7.5.6")),
            CompetencyReq("QA_CAPA", "ISO 14971 risk management and CAPA execution", "P", 2, False, ("ISO 14971", "CAPA", "clause 8.5.2")),
            CompetencyReq("QA_TACIT", "Cross-departmental audit composure and ethical vigilance", "V", 1, False, ()),
        ],
        "QMS": [
            CompetencyReq("QMS_CERT", "Certified ISO 13485 internal/lead auditor", "M", 3, True, ("ISO 13485", "lead auditor", "internal audit", "certificate")),
            CompetencyReq("QMS_LEAD", "Led ISO 13485 internal audits and NMPA QMS inspections", "M", 3, False, ("NMPA", "QMS inspection", "internal audit", "Order 739")),
            CompetencyReq("QMS_EQUIV", "Transferable ISO 9001 QMS maintenance with medical device alignment", "P", 2, False, ("ISO 9001", "ISO 13485", "equivalence", "QMS")),
            CompetencyReq("QMS_TACIT", "Regulatory inspection composure and ethical assertiveness", "V", 1, False, ()),
        ],
        "RA": [
            CompetencyReq("RA_DEG", "Bachelor degree in medicine, pharmacy, or life sciences", "M", 1, False, ("medicine", "pharmacy", "qualification")),
            CompetencyReq("RA_REG", "Independent Class II/III NMPA initial registration leadership", "M", 3, False, ("NMPA", "Class II", "Class III", "initial registration", "Order 797")),
            CompetencyReq("RA_CER", "Clinical Evaluation Report (CER) drafting and deficiency response", "P", 2, False, ("clinical evaluation", "CER", "deficiency response", "NMPA")),
            CompetencyReq("RA_TACIT", "Regulatory reviewer communication diplomacy and resilience", "V", 1, False, ()),
        ],
    }

    @classmethod
    def get_requirements(cls, role: str) -> List[CompetencyReq]:
        return cls.ONTOLOGY[role]


class OntologySchemaMasker:
    """
    Compiles strict JSON Schema constraints for structured decoding:
      - Tier M and P slots are exposed with closed status enum Sigma.
      - Tier V (tacit) slots are HARD-MASKED out of the LLM generation schema
        so the model is structurally unable to emit tacit-trait extrapolations.
    """

    @staticmethod
    def compile_generation_schema(role: str) -> dict:
        reqs = CompetencyOntology.get_requirements(role)
        extractable_ids = [r.req_id for r in reqs if r.tier in ("M", "P")]
        masked_ids = [r.req_id for r in reqs if r.tier == "V"]
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["candidate_id", "role", "entries"],
            "masked_tacit_slots": masked_ids,
            "properties": {
                "candidate_id": {"type": "string"},
                "role": {"type": "string", "enum": [role]},
                "entries": {
                    "type": "array",
                    "minItems": len(extractable_ids),
                    "maxItems": len(extractable_ids),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["req_id", "evidence", "anchor_lines", "status"],
                        "properties": {
                            "req_id": {"type": "string", "enum": extractable_ids},
                            "evidence": {"type": "string"},
                            "anchor_lines": {
                                "type": "array",
                                "items": {"type": "integer", "minimum": 1},
                                "minItems": 0,
                                "maxItems": 2,
                            },
                            "status": {"type": "string", "enum": list(CLOSED_STATUS_ENUM)},
                        },
                    },
                },
            },
        }

    @staticmethod
    def enforce_mask_and_validate(raw_card: dict, role: str) -> Tuple[dict, List[str]]:
        """Validate schema conformance and strip any forbidden tacit or scalar score fields."""
        violations: List[str] = []
        reqs = {r.req_id: r for r in CompetencyOntology.get_requirements(role)}
        allowed_keys = {"candidate_id", "role", "entries"}

        # Detect forbidden scalar scoring / ranking keys
        for k in list(raw_card.keys()):
            if k not in allowed_keys:
                violations.append(f"FORBIDDEN_FIELD_STRIPPED:{k}")

        clean_entries: List[dict] = []
        seen_ids = set()
        for entry in (raw_card.get("entries") or []):
            rid = entry.get("req_id", "")
            if rid not in reqs:
                violations.append(f"UNKNOWN_REQ_ID:{rid}")
                continue
            req = reqs[rid]
            ev_str = str(entry.get("evidence") or "")
            claim_raw = entry.get("claim_text")
            if req.tier == "V":
                # Hard-mask tacit extrapolation if an unmasked model tried to emit it
                if (
                    entry.get("status") not in (None, "", "blank")
                    or ev_str.strip()
                    or str(claim_raw or "").strip()
                ):
                    violations.append(f"TACIT_EXTRAPOLATION_BLOCKED:{rid}")
                continue
            if rid in seen_ids:
                violations.append(f"DUPLICATE_REQ_ID:{rid}")
                continue
            status = entry.get("status", "not_provided")
            if status not in CLOSED_STATUS_ENUM:
                violations.append(f"INVALID_ENUM_STATUS:{rid}:{status}")
                status = "conflicting"
            seen_ids.add(rid)
            claim_str = ev_str if claim_raw is None else str(claim_raw)
            raw_al = entry.get("anchor_lines")
            anchor_lines = list(raw_al) if isinstance(raw_al, (list, tuple)) else []
            clean_entries.append({
                "req_id": rid,
                "tier": req.tier,
                "evidence": ev_str,
                "anchor_lines": anchor_lines,
                "status": status,
                "claim_text": claim_str,
            })

        # Append structurally blank entries for Tier V (sigma = empty set)
        for req in CompetencyOntology.get_requirements(role):
            if req.tier == "V":
                clean_entries.append({
                    "req_id": req.req_id,
                    "tier": "V",
                    "evidence": "",
                    "anchor_lines": [],
                    "status": None,  # Formally empty set (blank)
                    "claim_text": "",
                })
            elif req.req_id not in seen_ids:
                violations.append(f"MISSING_REQ_FILLED_NOT_PROVIDED:{req.req_id}")
                clean_entries.append({
                    "req_id": req.req_id,
                    "tier": req.tier,
                    "evidence": "",
                    "anchor_lines": [],
                    "status": "not_provided",
                    "claim_text": "",
                })

        return {
            "candidate_id": raw_card.get("candidate_id", "UNKNOWN"),
            "role": role,
            "entries": clean_entries,
        }, violations


# =============================================================================
# 2. Version-Pinned Hybrid Regulatory RAG (BM25 + Normalized Cosine Similarity)
# =============================================================================

@dataclass(frozen=True)
class RegulatoryClause:
    doc_id: str
    version_id: str
    pinned: bool
    title: str
    text: str


DEFAULT_REGULATORY_CORPUS: List[RegulatoryClause] = [
    RegulatoryClause("ORD797-A35", "2025-01-v1", True, "Order 739/797 Art. 35 QMS Personnel",
                     "Medical device manufacturers shall establish ISO 13485 quality management systems staffed with qualified QA and regulatory personnel bearing statutory responsibility."),
    RegulatoryClause("ORD797-A86", "2025-01-v1", True, "Order 739/797 Art. 86 Compliance Liability",
                     "Production without qualified compliance personnel or non-conforming QMS triggers administrative sanctions, recalls, and license revocation."),
    RegulatoryClause("ISO13485-6.2", "2016-v1", True, "ISO 13485 Clause 6.2 Human Resources Competence",
                     "Personnel performing work affecting product quality shall be competent on the basis of appropriate education, training, skills, and verifiable experience records."),
    RegulatoryClause("ISO13485-7.5.6", "2016-v1", True, "ISO 13485 Clause 7.5.6 Validation of Processes",
                     "Special processes including sterilization, cleanroom control, and IQ OQ PQ equipment validation require documented protocol qualification and sterile barrier testing."),
    RegulatoryClause("ISO13485-8.2.4", "2016-v1", True, "ISO 13485 Clause 8.2.4 Internal Audit Qualification",
                     "Internal audits shall be conducted by certified independent auditors; mere audit preparation or training attendance does not substitute for certified internal auditor credentials without formal approval."),
    RegulatoryClause("ISO13485-8.5.2", "2016-v1", True, "ISO 13485 Clause 8.5.2 Corrective Action CAPA",
                     "Corrective and preventive action CAPA requires root cause investigation, ISO 14971 risk analysis updates, and effectiveness verification."),
    RegulatoryClause("SOP-EQ-01", "2025-01-v1.2", True, "SOP Cross-Industry Equivalence GMP to Medical Device",
                     "Sterile pharmaceutical GMP validation and precision electronics ISO 9001 internal auditor leadership may qualify as preferred equivalence subject to department manager escalation review."),
    RegulatoryClause("SOP-RA-02", "2025-01-v1.2", True, "NMPA Class II/III Registration and CER Guideline",
                     "Initial Class II and Class III NMPA medical device registration requires clinical evaluation report CER drafting, deficiency response handling, and current Order 739/797 regulatory compliance."),
    RegulatoryClause("SOP-CERT-03", "2025-01-v1.2", True, "Credential Verification Three-Element SOP",
                     "Mandatory statutory certificates require verifiable certificate name, certificate number or credential proof, and validity scope; training attendance alone triggers conflicting escalation."),
    RegulatoryClause("NMPA-GMP-STERILE", "2025-01-v1", True, "NMPA GMP Annex 1 Sterile Medical Devices",
                     "Sterile medical device manufacturing under NMPA GMP requires classified cleanroom control, microbial environmental monitoring, EO residual limits, and validated aseptic packaging."),
    RegulatoryClause("NMPA-GMP-IMPLANT", "2025-01-v1", True, "NMPA GMP Annex 2 Implantable Medical Devices",
                     "Implantable device manufacturers shall maintain full batch traceability, cleanroom assembly controls, biocompatibility verification, and qualified supplier audits."),
    RegulatoryClause("SOP-VAL-04", "2025-01-v1.2", True, "Internal SOP Equipment and Re-Validation Control",
                     "Manufacturing line transfers, software updates, and sterilization chamber modifications require formal IQ OQ PQ re-qualification protocols approved by QA engineering."),
    RegulatoryClause("ORD650-DEP", "2014-deprecated", False, "Deprecated 2014 Order 650 Guideline",
                     "Superseded medical device supervision rules prior to Order 739 and Order 797; must be excluded by version-pinned whitelist."),
]


class HybridRegulatoryRAG:
    """
    Hybrid sparse Okapi BM25 + sparse character 3-gram & token TF-IDF cosine
    retriever restricted strictly to version-pinned statutory clauses D_pin
    (unless pinned_only=False for ablation), executing on host CPU with zero
    extra GPU VRAM.
    """

    def __init__(
        self,
        corpus: Optional[List[RegulatoryClause]] = None,
        alpha: float = 0.5,
        k1: float = 1.5,
        b: float = 0.75,
        pinned_only: bool = True,
    ):
        self.all_docs = DEFAULT_REGULATORY_CORPUS if corpus is None else list(corpus)
        self.pinned_only = pinned_only
        self.pinned_docs = [d for d in self.all_docs if d.pinned] if pinned_only else list(self.all_docs)
        self.alpha = alpha
        self.k1 = k1
        self.b = b
        self._build_indices()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())

    @staticmethod
    def _char_ngrams(text: str, n: int = 3) -> List[str]:
        clean = re.sub(r"\s+", " ", text.lower()).strip()
        if len(clean) < n:
            return [clean] if clean else []
        return [clean[i : i + n] for i in range(len(clean) - n + 1)]

    def _build_indices(self) -> None:
        self.doc_tokens = [self._tokenize(d.title + " " + d.text) for d in self.pinned_docs]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.n_docs = len(self.pinned_docs)

        # BM25 IDF
        df: Counter = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                df[t] += 1
        self.idf = {
            t: math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for t, freq in df.items()
        }

        # Subword/token TF-IDF vectors for dense-proxy cosine similarity
        self.doc_ngrams = [
            Counter(self._tokenize(d.title + " " + d.text) + self._char_ngrams(d.title + " " + d.text, 3))
            for d in self.pinned_docs
        ]
        ndf: Counter = Counter()
        for cnt in self.doc_ngrams:
            for g in cnt:
                ndf[g] += 1
        self.ng_idf = {g: math.log((self.n_docs + 1) / (f + 1)) + 1.0 for g, f in ndf.items()}
        self.doc_vecs = [self._tfidf_vec(cnt) for cnt in self.doc_ngrams]

    def _tfidf_vec(self, counts: Counter) -> Dict[str, float]:
        vec = {k: (1.0 + math.log(v)) * self.ng_idf.get(k, 1.0) for k, v in counts.items() if v > 0}
        norm = math.sqrt(sum(val * val for val in vec.values())) or 1.0
        return {k: val / norm for k, val in vec.items()}

    def _bm25_scores(self, q_tokens: List[str]) -> List[float]:
        scores = []
        for toks, dl in zip(self.doc_tokens, self.doc_lens):
            tf = Counter(toks)
            score = 0.0
            for qt in q_tokens:
                if qt not in tf:
                    continue
                f = tf[qt]
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                score += self.idf.get(qt, 0.0) * (f * (self.k1 + 1.0)) / denom
            scores.append(score)
        max_s = max(scores) if scores else 0.0
        return [s / max_s if max_s > 0 else 0.0 for s in scores]

    def _cosine_scores(self, query: str) -> List[float]:
        """Corpus max-normalized token/3-gram TF-IDF cosine similarity over indexed corpus (matches BM25 [0, 1] range in Eq. 1)."""
        q_cnt = Counter(self._tokenize(query) + self._char_ngrams(query, 3))
        q_vec = self._tfidf_vec(q_cnt)
        scores = []
        for d_vec in self.doc_vecs:
            dot = sum(q_vec[k] * d_vec[k] for k in q_vec if k in d_vec)
            scores.append(dot)
        max_s = max(scores) if scores else 0.0
        return [s / max_s if max_s > 0 else 0.0 for s in scores]

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[RegulatoryClause, float]]:
        q_tokens = self._tokenize(query)
        bm25_norm = self._bm25_scores(q_tokens)
        cos_norm = self._cosine_scores(query)
        ranked = []
        for doc, s_bm25, s_cos in zip(self.pinned_docs, bm25_norm, cos_norm):
            s_hyb = self.alpha * s_bm25 + (1.0 - self.alpha) * s_cos
            ranked.append((doc, s_hyb))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# =============================================================================
# 3. Deterministic Span-Anchor & Qualification-Conflict Verifier
# =============================================================================

class DeterministicVerifier:
    """
    Zero-stochasticity post-extraction verifier executing between Stage 2 (LLM)
    and Stage 3 (Human Decision Gate):
      (a) Verbatim Span-Anchor Alignment Check
      (b) Qualification-Verb Escalation Guard (Weak -> Strong hallucination catch)
      (c) Statutory Credential Proof Check (Training-only vs. Certified)
      (d) Tier-V Tacit Slot Blank Invariant Check
    """

    WEAK_VERB_PATTERNS = [
        r"\bparticipat(?:ed|ing|e)\b",
        r"\bassist(?:ed|ing|ant)?\b",
        r"\bsupport(?:ed|ing)?\b",
        r"\bprepar(?:ation|ations|ing|ed)\b",
        r"\btrain(?:ing|ed)\b",
        r"\battend(?:ed|ing)\b",
        r"\bfamiliar(?:ity)?\b",
        r"参与", r"协助", r"支持", r"参加.*培训", r"培训", r"迎审准备", r"资料整理", r"助理",
    ]
    MODERATE_VERB_PATTERNS = [
        r"\bexecut(?:ed|ing)\b",
        r"\bperform(?:ed|ing)\b",
        r"\bmaintain(?:ed|ing)\b",
        r"\bdraft(?:ed|ing)\b",
        r"\bresponsib(?:le|ility)\b",
        r"\bhandl(?:ed|ing)\b",
        r"\bupdat(?:ed|ing)\b",
        r"\bclos(?:ed|ing)\b",
        r"负责", r"执行", r"撰写", r"维护", r"处理",
    ]
    _NON_UN_CERT_LB = r"(?<!\bnon-)(?<!\bnon )(?<!\bun-)(?<!\bun )(?<!\bself-)(?<!\bself )(?<!\bnot-)"
    _TRAINING_CERT_LB = (
        r"(?<!\btraining\s)(?<!\battendance\s)(?<!\bcompletion\s)(?<!\bparticipation\s)(?<!\bworkshop\s)(?<!\bcourse\s)"
    )

    _TEMPORAL_INVAL_ADV = (
        r"(?:(?:later|subsequently|recently|previously|ultimately|still|currently|presently|now|temporarily)\s+)?"
    )
    _COPULA_OPT = (
        r"(?:(?:is|was|are|were|remain(?:s|ed)?|(?:has|have|had)(?:\s+(?:been|remained))?)\s+)?"
    )
    _INVAL_SUBJECT_NOUNS = (
        r"(?:(?:audit|validation|capa|cer|inspection|test)\s+reports?"
        r"|(?:registration|nmpa|licen[cs]e|certif(?:ication|icate))\s+applications?"
        r"|\bcers?\b|\bcapas?\b|\biq(?:/oq(?:/pq)?)?\b|\boq/pq\b"
        r"|dossiers?|validations?|protocols?|audits?(?:\s+approvals?)?|reviews?|degrees?|diplomas?"
        r"|certif(?:icates?|ications?)|certs?(?:\.|(?=\s|$|[:;,.)\]）]))|credentials?|licen[cs]es?"
        r"|registrations?|approvals?|authorizations?|renewals?|verifications?|submissions?|filings?|records?)"
    )
    _INVAL_REJECT_STATES = r"(?:rejected|denied)(?:\s+by\s+[a-z0-9-]+)?"
    _NEG_POS_STATES = (
        r"(?:awarded|conferred|completed|closed|issued|obtained|verified|approved|renewed|passed"
        r"|valid|current|active|provided|held|listed|submitted|filed|granted|accepted|certified"
        r"|in\s+force|effective)"
    )
    _POS_STATE_ADV_OPT = (
        r"(?:(?:yet|still|currently|presently|now|formally|officially|fully|properly|independently|effectively|successfully)\s+){0,2}"
    )
    _AUX_NEGATED_POS_STATE = (
        rf"(?:(?:(?:is|was|are|were|remain(?:s|ed)?)\s+)?(?:(?:still|currently|presently|now)\s+)?not\s+{_POS_STATE_ADV_OPT}{_NEG_POS_STATES}"
        rf"|(?:isn't|aren't|wasn't|weren't)\s+{_POS_STATE_ADV_OPT}{_NEG_POS_STATES}"
        rf"|(?:(?:has|have|had)\s+not|hasn't|haven't|hadn't)\s+{_POS_STATE_ADV_OPT}been\s+{_POS_STATE_ADV_OPT}{_NEG_POS_STATES}"
        rf"|(?:has|have|had|is|was|are|were)\s+yet\s+to\s+be\s+{_POS_STATE_ADV_OPT}{_NEG_POS_STATES}"
        rf"|(?:cannot|can\s+not|can't|(?:could|would|should|will|may|might)\s+not|couldn't|wouldn't|shouldn't|won't)\s+{_POS_STATE_ADV_OPT}be\s+{_POS_STATE_ADV_OPT}{_NEG_POS_STATES})"
    )
    _INVAL_LIFECYCLE_STATES = (
        rf"(?:pending(?:\s+approval)?|incomplete|in\s+progress"
        rf"|{_AUX_NEGATED_POS_STATE}"
        rf"|unapproved|unclosed|unissued|unconfirmed|unsubmitted|unfinished|uncompleted|unrenewed|unaccepted|ungranted"
        rf"|revoked|suspended|withdrawn|withheld|expired|lapsed|invalid(?:ated)?|void(?:ed)?|cancell?ed|failed)"
    )
    _NO_LONGER_INVAL = r"no\s+longer\s+(?:valid|active|current|certified|held|in\s+force|effective)"

    _LICENSE_INVAL_LA = (
        rf"(?!\s*(?:[:：]\s*|\s+){_COPULA_OPT}{_TEMPORAL_INVAL_ADV}(?:{_INVAL_LIFECYCLE_STATES}|{_INVAL_REJECT_STATES}|{_NO_LONGER_INVAL})\b)"
    )

    STRONG_CERT_PATTERNS = [
        rf"{_NON_UN_CERT_LB}{_TRAINING_CERT_LB}\bcertif(?:ied|icate|ication)(?!\s+of\s+(?:[a-z0-9-]+\s+){{0,5}}(?:training|attendance|completion|participation)\b)\b",
        rf"{_NON_UN_CERT_LB}\blicens(?:ed|e)\b{_LICENSE_INVAL_LA}",
        r"\blead\s+auditor\b",
        r"\bled\b",
        r"\bdirect(?:ed|ing)\b",
        r"\bindependent(?:ly)?\b",
        rf"{_NON_UN_CERT_LB}{_TRAINING_CERT_LB}\bcert(?:ificate)?\.?\s*(?:no\.?|id|#)\b",
        r"(?<![未非无不曾])(?<!不再)(?<!未正式)(?<!未实际)(?<!未有效)持证", r"证书编号", r"内审员证", r"主任审核员", r"主导", r"独立申报", r"独立主导",
    ]
    _STRONG_CERT_CRED_PATTERNS = [
        rf"{_NON_UN_CERT_LB}{_TRAINING_CERT_LB}\bcertif(?:ied|icate|ication)(?!\s+of\s+(?:[a-z0-9-]+\s+){{0,5}}(?:training|attendance|completion|participation)\b)\b",
        rf"{_NON_UN_CERT_LB}\blicens(?:ed|e)\b{_LICENSE_INVAL_LA}",
        r"\blead\s+auditor\b",
        rf"{_NON_UN_CERT_LB}{_TRAINING_CERT_LB}\bcert(?:ificate)?\.?\s*(?:no\.?|id|#)\b",
        r"(?<![未非无不曾])(?<!不再)(?<!未正式)(?<!未实际)(?<!未有效)持证", r"证书编号", r"内审员证", r"主任审核员",
    ]
    _STRONG_CERT_ACTION_PATTERNS = [
        r"\bled\b",
        r"\bdirect(?:ed|ing)\b",
        r"\bindependent(?:ly)?\b",
        r"主导", r"独立申报", r"独立主导",
    ]
    CERT_PROOF_PATTERNS = [
        rf"{_NON_UN_CERT_LB}{_TRAINING_CERT_LB}\bcert(?:if(?:ied|icate))?\.?\s*(?:no\.?|id|#|number)\b",
        rf"{_NON_UN_CERT_LB}\bcertified\s+(?:iso\s*13485(?:\s+(?:internal|lead))?|internal|lead)\s+auditor\b",
        r"\bcert-[a-z0-9-]+\b",
        r"(?<![未非无不曾])(?<!不再)(?<!未正式)(?<!未实际)(?<!未有效)持证", r"证书编号", r"内审员证", r"资格证书",
    ]
    TRAINING_ONLY_PATTERNS = [
        r"\btrain(?:ing|ed)\b",
        r"\bworkshop\b",
        r"\bcourse\b",
        r"参加.*培训", r"培训结业",
    ]

    _WEAK_VERB_RES = [re.compile(p) for p in WEAK_VERB_PATTERNS]
    _MODERATE_VERB_RES = [re.compile(p) for p in MODERATE_VERB_PATTERNS]
    _STRONG_CERT_RES = [re.compile(p) for p in STRONG_CERT_PATTERNS]
    _STRONG_CERT_CRED_RES = [re.compile(p) for p in _STRONG_CERT_CRED_PATTERNS]
    _STRONG_CERT_ACTION_RES = [re.compile(p) for p in _STRONG_CERT_ACTION_PATTERNS]
    _CERT_PROOF_RES = [re.compile(p) for p in CERT_PROOF_PATTERNS]
    _TRAINING_ONLY_RES = [re.compile(p) for p in TRAINING_ONLY_PATTERNS]

    _STRIP_PUNCT_CHARS = frozenset("，。；：、（）()[]\"'“”‘’")
    _SENT_SPLIT_RE = re.compile(
        r"[;!?。；！？\n]+|(?<!\bcert)(?<!\bno)(?<!\bv)(?<!\bart)(?<!\bfig)(?<!\bref)(?<!\brev)\.(?:\s|$)"
        r"|(?<=\bcert)\.(?:\s+(?!no\.?\b|id\b|#|number\b)|$)",
        re.IGNORECASE,
    )
    _CLAUSE_SPLIT_RE = re.compile(
        r"[;!?。；！？,，\n]+|(?<!\bcert)(?<!\bno)(?<!\bv)(?<!\bart)(?<!\bfig)(?<!\bref)(?<!\brev)\.(?:\s|$)"
        r"|(?<=\bcert)\.(?:\s+(?!no\.?\b|id\b|#|number\b)|$)",
        re.IGNORECASE,
    )
    _COMMA_SPLIT_RE = re.compile(r"[,，]+")

    # Quality/compliance outcome whitelists (prevent FP on "no nonconformity", "无偏差", "无菌", etc.)
    _EN_QUAL_LA = (
        r"(?!\s+(?:(?!certif|cert\b|credent|experienc|degree|verif|proof|evidenc|qualif|train|auditor|licens)[a-z0-9-]+\s+){0,3}"
        r"(?:non-?conformit(?:y|ies)|ncr?s?|deviation(?:s)?|defect(?:s)?|finding(?:s)?|observation(?:s)?"
        r"|incident(?:s)?|recall(?:s)?|adverse\s+event(?:s)?|complaint(?:s)?|contamination|error(?:s)?|overdue|backlog)\b)"
    )
    _WITHOUT_AUTONOMY_LA = (
        r"(?!\s+(?:direct\s+|close\s+|constant\s+|external\s+|outside\s+|senior\s+|third-party\s+|3rd-party\s+)?(?:supervision|oversight|assistance|support|prompting|guidance|intervention)\b)"
    )
    _CN_QUAL_LA = (
        r"(?!(?:任何|明显|显著)?(?:重大|严重|一般)?(?:未关闭|待关闭|未整改|遗留|未闭环|未解决)?(?:的)?(?:医疗器械|器械|质量|安全|临床|生产|合规|审计|审核|监管|gmp|iso)?(?:菌|尘|毒|害|损|缺陷|不符合|偏差|异常|不良|投诉|召回|污染|逾期|发补逾期|差错|事故|事件|故障|发现项?|违[规纪]))"
    )
    _NO_NEG = (
        r"(?<!certificate\s)(?<!cert\s)(?<!cert\.\s)(?<!license\s)(?<!id\s)(?<!reg\.\s)(?<!registration\s)"
        rf"\bno(?!\.)\b{_EN_QUAL_LA}"
    )
    _WITHOUT_NEG = rf"\bwithout\b{_EN_QUAL_LA}{_WITHOUT_AUTONOMY_LA}"
    _STRICT_ZERO_EXP_NEG = (
        r"(?:(?:(?<![a-z0-9.])(?:0(?:\.0+)?|\.0+)(?![\d.])|\bzero)[\s-]*"
        r"(?:(?:years|year|yrs|yr|months|month|mos|mo)\b|年|个月|(?:relevant\s+|prior\s+|medical\s+device\s+|qa\s+|qc\s+|ra\s+|qms\s+){0,4}experience\b)"
        r"|(?:<\s*|\b(?:less\s+than|fewer\s+than|under|below)\s+)(?:(?:1(?:\.0+)?|one)\s*(?:years|year|yrs|yr)\b|(?:[1-9]|1[012]|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(?:months|month|mos|mo)\b)"
        r"|(?:不足|不到|未满|少于)\s*(?:(?:1|一)\s*年|(?:[1-9]|1[012]|一|二|两|三|四|五|六|七|八|九|十|十一|十二)\s*个?月))"
    )
    _SUB_YEAR_EXP_NEG = (
        r"(?:(?<![a-z0-9.])0?\.(?!0+(?![\d.]))[0-9]+(?![\d.])[\s-]*(?:(?:years|year|yrs|yr|months|month|mos|mo)\b|年|个月)"
        r"|(?<![a-z0-9.])(?<!\bin\s)(?<!\band\s)(?<!\bafter\s)(?<!\byears\s)(?<!\byears,\s)(?<!\byrs\s)(?<!\byear\s)(?<!\bevery\s)(?<!\bfor\s)(?<!\bwithin\s)(?<!\bpast\s)(?<!\blast\s)(?<!\bover\s)(?<!\bunder\s)(?<!\bthan\s)(?<!年)(?:1[01]|[1-9])(?![\d.])[\s-]*"
        r"(?:(?:months|month|mos|mo)\b(?!(?:['’]s?)?(?:\s+[a-z0-9-]+){0,2}\s+(?:ago|later|prior|before|after|exchange|internship|course|training|workshop|seminar)\b)"
        r"|个月(?!\s*(?:实习|培训|课程|交换|轮岗|研讨会|见习|进修))))"
    )
    _QA_EXP_SUB_YEAR_NEG = (
        r"(?:(?<![a-z0-9.])0?\.(?!0+(?![\d.]))[0-9]+(?![\d.])\+?[\s-]*(?:(?:years|year|yrs|yr|months|month|mos|mo)\b|年|个月)"
        r"|(?<![a-z0-9.])(?<!\byears\s)(?<!\byears,\s)(?<!\byrs\s)(?<!\byrs,\s)(?<!\byear\s)(?<!\byear,\s)(?<!\byr\s)"
        r"(?<!\byears\sand\s)(?<!\byrs\sand\s)(?<!\byear\sand\s)(?<!\byr\sand\s)"
        r"(?<!\bevery\s)(?<!\bwithin\s)(?<!\bvalid\sfor\s)(?<!年)(?<!年零)"
        r"(?:1[01]|[1-9]|one|two|three|four|five|six|seven|eight|nine|ten|eleven"
        r"|(?<![一二两三四五六七八九十廿百])(?:十一|十|一|二|两|三|四|五|六|七|八|九)(?![一二三四五六七八九十]))"
        r"(?![a-z0-9.])\+?[\s-]*"
        r"(?:(?:months|month|mos|mo)\b(?!(?:['’]s?)?(?:\s+[a-z0-9-]+){0,2}\s+(?:exchange|internship|course|training|workshop|seminar)\b)"
        r"|个月(?!\s*(?:实习|培训|课程|交换|轮岗|研讨会|见习|进修))))"
    )
    _ZERO_EXP_NEG = _STRICT_ZERO_EXP_NEG
    _SUB_YEAR_EXP_RE = re.compile(_SUB_YEAR_EXP_NEG)
    _QA_EXP_SUB_YEAR_RE = re.compile(_QA_EXP_SUB_YEAR_NEG)
    _QUALIFYING_MULTI_YEAR_RE = re.compile(
        r"(?:(?<![a-z0-9.])(?!(?:19|20)\d{2}\+?[\s-]*(?:(?:years|year|yrs|yr)\b|年))[1-9]\d*(?:\.\d+)?\+?[\s-]*(?:(?:years|year|yrs|yr)\b|年)"
        r"|(?<![a-z0-9.])(?:1[2-9]|[2-9]\d+)\+?[\s-]*(?:(?:months|month|mos|mo)\b|个月))"
    )
    _NON_UN_PREFIX_NEG = (
        r"(?:\b(?:non|un|self)[\s-]+(?:certif(?:ied|icate|ication)|licen[cs]ed?|verified|accredited|valid(?:ated)?)\b"
        r"|\b(?:uncertified|unlicensed|unaccredited)\b)"
    )
    _PRE_INVAL_MOD_NEG = (
        r"\b(?:expired|revoked|suspended|invalid(?:ated)?|void(?:ed)?|cancell?ed|lapsed|withdrawn|pending|unverified)"
        r"\s+(?:[a-z0-9-]+\s+){0,5}"
        r"(?:certif(?:ied|icates?|ications?)\b|certs?(?:\.|(?=\s|$|[,;.)\]）]))|licen[cs]es?|auditors?|credentials?|registrations?|approvals?|authorizations?|renewals?|submissions?|filings?)"
    )
    _PLACEHOLDER_CERT_NEG = (
        r"(?:\bcert(?:if(?:ied|icate))?\.?\s*(?:no\.?|id|#|number)\s*[:：.]?\s*[\[(（【]?\s*"
        r"(?:(?:n/?a|none|null|tbd|pending|missing|expired|revoked|invalid|void(?:ed)?|cancell?ed|lapsed|withdrawn)\b"
        r"|(?:无|暂无|待补|待定|待发|缺失)(?![a-z0-9\u4e00-\u9fff])"
        r"|[-—]+(?=[\])）】]?(?:\s|$|[,;.)\]），。；])))"
        r"|(?:证书编号|证书号)\s*[:：]?\s*[\[(（【]?\s*"
        r"(?:(?:n/?a|none|null|tbd|pending|missing|expired|revoked|invalid|void(?:ed)?|cancell?ed|lapsed|withdrawn)\b"
        r"|(?:无|暂无|待补|待定|待发|缺失)(?![a-z0-9\u4e00-\u9fff])"
        r"|[-—]+(?=[\])）】]?(?:\s|$|[,;.)\]），。；]))))"
    )
    _EXPECTED_MONTH_OPT = (
        r"(?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?"
        r"|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
        r"|spring|summer|fall|autumn|winter)\s+)?"
    )
    _EXPECTED_YEAR_PAT = (
        rf"(?:\(\s*expected(?:\s*[:：])?(?:\s+(?:graduation|completion))?(?:\s+in)?\s+{_EXPECTED_MONTH_OPT}(?:19|20)\d{{2}}\s*\)"
        rf"|(?:,\s*|;\s*|\b)expected(?:\s*[:：])?(?:\s+(?:graduation|completion))?(?:\s+in)?\s+{_EXPECTED_MONTH_OPT}(?:19|20)\d{{2}}\b)"
    )
    _CN_INVAL_PHRASES = (
        r"(?:被驳回|已驳回|遭到?驳回|被拒绝|已拒绝|被退回|已退回"
        r"|未获批|尚未获批|暂未获批|未批准|尚未批准|暂未批准|仍未批准"
        r"|被撤销|已撤销|已撤回|被撤回|被暂停|已暂停|被吊销|已吊销|被作废|已作废|被注销|已注销"
        r"|未生效|尚未生效|暂未生效|注册失败|未通过|尚未通过|暂未通过|仍未通过"
        r"|未完成|尚未完成|暂未完成|仍未完成|未提交|尚未提交|暂未提交|仍未提交"
        r"|未签发|尚未签发|暂未签发|仍未签发|待审批|待审核|待批准|待关闭|审批中|审核中"
        r"|已过期|已失效|不再有效|待取证|备考中)"
    )
    _CN_INVAL_SUBJECTS = (
        r"(?:(?:审计|审核|验证|检验|测试|检测|注册|申报|申请|资质|授权|质量|合规)?\s*"
        r"(?:报告|档案|资料|记录|方案|申请|注册|申报|审批|审核|审计|资质|授权|证书|执照|学历|学位|验证)"
        r"|capa|cer|iq(?:/oq(?:/pq)?)?)"
    )
    _CN_SUBJ_INVAL = (
        rf"(?<!无)(?<!零)(?<!没有)(?<!无任何)(?<!不存在)"
        rf"{_CN_INVAL_SUBJECTS}\s*[:：]?\s*"
        r"(?:无效|未关闭|尚未关闭|暂未关闭|仍未关闭|不完整|不合格)"
    )
    _VERIF_FAIL_RE = re.compile(
        rf"{_NO_NEG}\s+(?:[a-z0-9-]+\s+){{0,2}}(?:verifications?|proofs?|evidence|formal\s+records?)\b"
        rf"|{_WITHOUT_NEG}\s+(?:[a-z0-9-]+\s+){{0,2}}(?:verifications?|proofs?|evidence|formal\s+records?)\b"
        r"|\b(?:(?:did|does|do|has|have|had|could|would|should|will|may|might)\s+not|cannot|can\s+not|didn't|doesn't|don't|hasn't|haven't|hadn't|couldn't|wouldn't|shouldn't|won't|can't|unable\s+to|failed\s+to)\s+"
        r"(?:obtain(?:ed)?|have|had|provide(?:d)?|pass(?:ed)?|receive(?:d)?|complete(?:d)?|earn(?:ed)?|finish(?:ed)?|close(?:d)?|submit(?:ted)?|renew(?:ed)?|hold|held|maintain(?:ed)?|verif(?:y|ied))\s+"
        r"(?:[a-z0-9-]+\s+){0,3}(?:verifications?|proofs?|evidence|formal\s+records?|reviews?|audits?|validations?|protocols?|dossiers?|reports?|cers?|capas?|degrees?|diplomas?|credentials?|licen[cs]es?|certif(?:icates?|ications?)|registrations?|approvals?|authorizations?|renewals?|submissions?|filings?)\b"
        r"|\b(?:lack(?:s|ed|ing)?|missing)\s+(?:[a-z0-9-]+\s+){0,2}(?:verifications?|proofs?|evidence|formal\s+records?)\b"
        rf"|\b(?:verifications?|approvals?|registrations?|authorizations?|renewals?|submissions?)\s+{_COPULA_OPT}{_TEMPORAL_INVAL_ADV}pending\b"
        r"|\bpending\s+(?:verifications?|approvals?|registrations?|authorizations?|renewals?)\b"
        r"|\b(?:unverified|not\s+(?:(?:yet|formally|independently)\s+){0,2}verified)\b"
        r"|\b(?:(?:did|does|has|have|had)\s+not|didn't|doesn't|hasn't|haven't|hadn't)\s+graduate(?:d)?\b"
        rf"|\b{_INVAL_SUBJECT_NOUNS}(?:\s*[:：]\s*|\s+){_COPULA_OPT}{_TEMPORAL_INVAL_ADV}{_INVAL_REJECT_STATES}\b"
        rf"|\b{_INVAL_SUBJECT_NOUNS}(?:\s*[:：]\s*|\s+){_COPULA_OPT}{_TEMPORAL_INVAL_ADV}(?:{_INVAL_LIFECYCLE_STATES}|{_NO_LONGER_INVAL})\b"
        rf"|{_EXPECTED_YEAR_PAT}"
        rf"|\bstatus(?:\s*[:：]\s*|\s+){_COPULA_OPT}{_TEMPORAL_INVAL_ADV}(?:{_INVAL_LIFECYCLE_STATES}|{_INVAL_REJECT_STATES}|unverified|{_NO_LONGER_INVAL})\b"
        rf"|未核实|未验证|无正式证明|未通过验证|{_CN_INVAL_PHRASES}|{_CN_SUBJ_INVAL}"
    )

    _LOCAL_PRE_NEG_RE = re.compile(
        rf"{_NO_NEG}|{_WITHOUT_NEG}|{_ZERO_EXP_NEG}|{_NON_UN_PREFIX_NEG}|{_PRE_INVAL_MOD_NEG}"
        r"|\b(?:not(?!\s+only\b)|never|neither|nor|lack(?:ed|ing|s)?|missing|unverified|cannot|(?:failed|unable|refused|declined|not\s+allowed)\s+to)\b"
        r"|[a-z]+n't\b"
        rf"|未写|无{_CN_QUAL_LA}|缺(?!陷)|未附|未持证|未持有|未取得|未持|未获得|未正式|未实际|未通过|非持证|不再持证|不具备|没有{_CN_QUAL_LA}|并未|未曾|尚未|暂未|无效.{{0,4}}证"
    )
    _PRE_AUTH_NEG_RE = re.compile(
        r"\b(?:not|never|without)[\s-]+(?:[a-z0-9-]+\s+){0,2}"
        r"(?:authorized|authorization|approved|approval|permitted|permission|qualified|qualification|allowed|eligible|authority|licensed|certified|capable|able)\b"
        r"|\b(?:unapproved|unauthorized|unpermitted|disapproved)\s+by\b"
        r"|未授权|无权|不具备.{0,4}资格|无资格|未获(?:批准|授权)|未经(?:批准|授权)"
    )
    _ANAPHORIC_AUTH_NEG_RE = re.compile(
        r"\b(?:not|never|without)[\s-]+(?:[a-z0-9-]+\s+){0,2}(?:approved|approval)\s+by\b"
        r"|\b(?:unapproved|unauthorized|unpermitted|disapproved)\s+by\b"
        r"|\b(?:not|never)[\s-]+(?:[a-z0-9-]+\s+){0,2}(?:authorized|permitted|allowed)\s+to\s+(?:do\s+so|perform|conduct|lead|execute)\b"
        r"|^\s*(?:not|never)[\s-]+(?:approved|authorized|permitted|allowed)\s*\.?\s*$"
        r"|^\s*without\s+(?:[a-z0-9-]+\s+){0,2}(?:approval|authorization|permission)\s*\.?\s*$"
        r"|未获(?:批准|授权)|未经(?:批准|授权)"
    )
    _STANDALONE_STATUS_SENT_RE = re.compile(
        r"^\s*(?:status(?:\s*[:：]\s*|\s+))?(?:"
        rf"{_COPULA_OPT}{_TEMPORAL_INVAL_ADV}(?:{_INVAL_LIFECYCLE_STATES}|{_INVAL_REJECT_STATES}|unverified|{_NO_LONGER_INVAL})(?:\s+in\s+\d{{4}}|\s+pending\s+[a-z0-9-]+)?"
        rf"|{_INVAL_SUBJECT_NOUNS}(?:\s*[:：]\s*|\s+){_COPULA_OPT}{_TEMPORAL_INVAL_ADV}(?:{_INVAL_LIFECYCLE_STATES}|{_INVAL_REJECT_STATES}|unverified|{_NO_LONGER_INVAL})"
        rf"|{_EXPECTED_YEAR_PAT}"
        rf"|(?<!无)(?<!零)(?<!没有)(?<!无任何)(?<!不存在){_CN_INVAL_SUBJECTS}\s*[:：]?\s*(?:{_CN_INVAL_PHRASES}|无效|未关闭|尚未关闭|暂未关闭|仍未关闭|不完整|不合格)"
        rf"|{_CN_INVAL_PHRASES}|无效"
        r")\s*\.?\s*$"
    )
    _WEAK_DISCLAIMER_RE = re.compile(
        r"\b(?:not|never|no|without)\s+(?:[a-z0-9-]+\s+){0,2}(?:trained|training|familiar|familiarity|exposed|exposure)\b"
    )
    _GENERIC_TOPIC_NOUNS = frozenset({
        "validation", "audit", "audits", "process", "processes", "system", "systems",
        "management", "testing", "control", "documentation", "records", "device", "devices",
        "quality", "medical", "engineer", "engineering",
    })
    _LOCAL_POST_NEG_RE = re.compile(
        rf"\b(?:{_AUX_NEGATED_POS_STATE}|unverified|lacking|missing)\b"
        r"|\bfailed\s+to\b"
        rf"|{_ZERO_EXP_NEG}"
        rf"|{_WITHOUT_NEG}\s+(?:[a-z0-9-]+\s+){{0,2}}(?:verifications?|proofs?|records?|certif(?:icates?|ications?)|approvals?|registrations?|authorizations?)\b"
        rf"|\(\s*{_TEMPORAL_INVAL_ADV}(?:{_INVAL_LIFECYCLE_STATES}|{_INVAL_REJECT_STATES}|unverified|{_NO_LONGER_INVAL})\s*\)"
        rf"|\b{_INVAL_SUBJECT_NOUNS}(?:\s*[:：]\s*|\s+){_COPULA_OPT}{_TEMPORAL_INVAL_ADV}(?:{_INVAL_LIFECYCLE_STATES}|{_INVAL_REJECT_STATES}|unverified|{_NO_LONGER_INVAL})\b"
        rf"|\b{_TEMPORAL_INVAL_ADV}(?:expired|revoked|suspended|invalidated|voided|cancelled|canceled|lapsed|withdrawn)\s+in\s+\d{{4}}\b"
        rf"|{_EXPECTED_YEAR_PAT}"
        rf"|(?:^\s*|,\s*|;\s*)(?:status(?:\s*[:：]\s*|\s+))?{_COPULA_OPT}{_TEMPORAL_INVAL_ADV}(?:{_INVAL_LIFECYCLE_STATES}|{_INVAL_REJECT_STATES}|unverified|{_NO_LONGER_INVAL})(?:\s+in\s+\d{{4}}|\s+pending\s+[a-z0-9-]+)?\b"
        rf"|{_PLACEHOLDER_CERT_NEG}"
        rf"|{_NO_NEG}\s+(?:[a-z0-9-]+\s+){{0,2}}(?:certif(?:icates?|ications?)\b|certs?(?:\.|(?=\s|$|[,;.)\]）]))|(?:credentials?|records?|proofs?|evidence|verifications?|registrations?|approvals?|authorizations?|issued|listed|provided|held)\b)"
        rf"|未附|未提供|未取得|未获得|未持证|未持有|无效.{{0,4}}证|证书.{{0,4}}无效|缺失|{_CN_INVAL_PHRASES}|{_CN_SUBJ_INVAL}"
    )
    _SPAN_PRE_NEG_RE = re.compile(
        r"(?:"
        r"(?<!certificate\s)(?<!cert\s)(?<!cert\.\s)(?<!license\s)(?<!id\s)(?<!reg\.\s)(?<!registration\s)\bno(?!\.)\s+longer(?:\s+[a-z0-9-]+){0,3}"
        rf"|{_NO_NEG}(?:\s+[a-z0-9-]+){{0,4}}"
        rf"|{_WITHOUT_NEG}(?:\s+[a-z0-9-]+){{0,4}}"
        rf"|{_ZERO_EXP_NEG}(?:\s+(?:of\s+)?[a-z0-9-]+){{0,4}}"
        rf"|{_NON_UN_PREFIX_NEG}(?:\s+[a-z0-9-]+){{0,3}}"
        rf"|{_PRE_INVAL_MOD_NEG}(?:\s+[a-z0-9-]+){{0,3}}"
        r"|\b(?:not(?!\s+only\b)|never|neither|nor|lack(?:ed|ing|s)?|missing|unverified|cannot|[a-z]+n't)\b(?:\s+[a-z0-9-]+){0,4}"
        r"|\b(?:failed|unable|refused|declined|not\s+allowed)\s+to(?:\s+[a-z0-9-]+){0,4}"
        r"|\b(?:did|does|do|has|have|had|was|were|is|are|can|could|will|would|should)\s+not(?!\s+only\b)(?:\s+[a-z0-9-]+){0,3}"
        rf"|(?:未写|无{_CN_QUAL_LA}|缺(?!陷)|未附|未持|未取得|未获得|不具备|没有|并未|未曾|尚未|暂未|不再).{{0,10}}"
        r")\s*$"
    )
    _EV_START_NEG_RE = re.compile(
        r"^(?:(?:not(?!\s+only\b)|never|neither|nor|lack(?:ed|ing|s)?|missing|unverified|cannot|(?:failed|unable|refused|declined|not\s+allowed)\s+to|did\s+not|[a-z]+n't)\b"
        rf"|{_NO_NEG}|{_WITHOUT_NEG}|{_ZERO_EXP_NEG}|{_NON_UN_PREFIX_NEG}|{_PRE_INVAL_MOD_NEG}"
        rf"|未写|无{_CN_QUAL_LA}|缺(?!陷)|未附|未持|未取得|未获得|未正式|未实际|未通过|非持证|不再持证|不具备|没有|并未|未曾|尚未|暂未)"
    )
    _INTERNAL_NEG_RE = re.compile(
        rf"{_NO_NEG}|{_WITHOUT_NEG}|{_ZERO_EXP_NEG}|{_NON_UN_PREFIX_NEG}|{_PRE_INVAL_MOD_NEG}|{_PLACEHOLDER_CERT_NEG}"
        r"|\b(?:not(?!\s+only\b)|never|neither|nor|lack(?:ed|ing|s)?|missing|unverified|cannot|(?:failed|unable|refused|declined|not\s+allowed)\s+to|did\s+not)\b"
        r"|[a-z]+n't\b"
        rf"|未写|无{_CN_QUAL_LA}|缺(?!陷)|未附|未持|未取得|未获得|未正式|未实际|非持证|不再持证|不具备|没有|并未|未曾|尚未|暂未|无效.{{0,4}}证|证书.{{0,4}}无效|{_CN_INVAL_PHRASES}|{_CN_SUBJ_INVAL}"
    )
    _CERT_NEG_RE = re.compile(
        rf"{_NON_UN_PREFIX_NEG}|{_PRE_INVAL_MOD_NEG}|{_PLACEHOLDER_CERT_NEG}|{_VERIF_FAIL_RE.pattern}"
        r"|\b(?:not|never|without|lack(?:ed|ing|s)?|missing|[a-z]+n't"
        r"|(?:did|does|do|has|have|had|is|was|were|are|can|could|will|would|should|may|might)\s+not"
        r"|(?:failed|unable|refused|declined)\s+to)"
        r"[\s-]+(?:[a-z0-9-]+\s+){0,6}(?:certif(?:ied|icates?|ications?)\b|certs?(?:\.|(?=\s|$|[,;.)\]）]))|credentials?\b|licen[cs]es?\b)"
        rf"|{_NO_NEG}\s+(?:(?!no\b)[a-z0-9-]+\s+){{0,5}}(?:certif(?:ied|icates?|ications?)\b|certs?(?:\.|(?=\s|$|[,;.)\]）]))|credentials?\b|licen[cs]es?\b)"
        rf"|\b{_INVAL_SUBJECT_NOUNS}(?:\s*[:：]\s*|\s+){_COPULA_OPT}{_TEMPORAL_INVAL_ADV}(?:{_INVAL_LIFECYCLE_STATES}|{_INVAL_REJECT_STATES}|unverified|missing|lacking|{_NO_LONGER_INVAL})\b"
        rf"|\b(?:certif(?:ied|icates?|ications?)\b|certs?(?:\.|(?=\s|$|[,;.)\]）]))|(?:auditors?|licen[cs]es?|credentials?)\b)[^.;\n]{{0,30}}\b{_TEMPORAL_INVAL_ADV}(?:{_NO_LONGER_INVAL}|expired|revoked|suspended|pending(?:\s+approval)?|unverified|missing|lacking|invalid(?:ated)?|void(?:ed)?|cancell?ed|lapsed|withdrawn|in\s+progress|failed)\b"
        rf"|(?:[;.,]\s*|^\s*)(?:status(?:\s*[:：]\s*|\s+))?{_COPULA_OPT}{_TEMPORAL_INVAL_ADV}(?:{_INVAL_LIFECYCLE_STATES}|{_INVAL_REJECT_STATES}|unverified|{_NO_LONGER_INVAL})(?:\s+in\s+\d{{4}}|\s+pending\s+[a-z0-9-]+)?\s*\.?\s*$"
        rf"|(?:（|\()\s*{_TEMPORAL_INVAL_ADV}(?:{_INVAL_LIFECYCLE_STATES}|{_INVAL_REJECT_STATES}|unverified|{_NO_LONGER_INVAL}|{_CN_INVAL_PHRASES}|无效)\s*(?:）|\))"
        r"|未写.{0,4}持证|未附.{0,4}证|未正式持证|未实际持证|(?:未|非|无|不|未曾|不再|尚未|暂未|并未).{0,6}持证"
        r"|未获得.{0,4}证|未取得.{0,4}证|未持有.{0,4}证|无有效.{0,4}证|无证书|缺证书"
        r"|证书.{0,6}(?:过期|失效|吊销|作废|注销|撤销|驳回|拒绝|暂停|未获批|尚未获批|暂未获批|未批准|尚未批准|暂未批准|无效|缺失|未附|未提供|未取得|未获得|未生效|尚未生效|暂未生效)"
        rf"|{_CN_INVAL_PHRASES}|{_CN_SUBJ_INVAL}|无效.{{0,4}}证|未通过.{{0,6}}(?:认证|证)"
    )
    _CERT_EXPLICIT_RE = re.compile(
        rf"{_NON_UN_CERT_LB}{_TRAINING_CERT_LB}cert(?:ificate)?\.?\s*(?:no\.?|id|#)"
        r"|cert-[a-z0-9-]+|(?<![未非无不曾])(?<!不再)(?<!未正式)(?<!未实际)(?<!未有效)持证|证书编号|内审员证"
    )
    _TRAINING_CERT_RE = re.compile(
        r"\b(?:training|attendance|completion|participation|workshop|course)\s+(?:(?!and\b|or\b|with\b|plus\b|then\b)[a-z0-9-]+\s+){0,4}certif(?:icate|ication)?\b"
        r"|\bcertif(?:icate|ication)\s+of\s+(?:[a-z0-9-]+\s+){0,5}(?:training|attendance|completion|participation)\b"
        r"|培训结业证|培训证书"
    )
    _STATUTORY_CERT_RE = re.compile(
        rf"{_NON_UN_CERT_LB}{_TRAINING_CERT_LB}\b(?:internal|lead)\s+auditor\s+certif(?:icate|ication)\b"
        rf"|{_NON_UN_CERT_LB}\bcertified\s+(?:iso\s*13485(?:\s+(?:internal|lead))?|internal|lead)\s+auditor\b"
        r"|(?<![未非无不曾])(?<!不再)(?<!未正式)(?<!未实际)(?<!未有效)持证|内审员证|资格证书"
    )
    _MULTI_SEG_SPLIT_RE = re.compile(r"[;；]|(?:\.{3}|…)")

    def __init__(
        self,
        span_threshold: float = 0.85,
        max_anchor_span: int = 3,
        use_fuzzy_span: bool = True,
        use_verb_tier: bool = True,
        use_cert_proof: bool = True,
    ):
        self.span_threshold = span_threshold
        self.max_anchor_span = max_anchor_span
        self.use_fuzzy_span = use_fuzzy_span
        self.use_verb_tier = use_verb_tier
        self.use_cert_proof = use_cert_proof

    @staticmethod
    def _normalize_text(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[\s\u3000]+", " ", s)
        s = re.sub(r"[，。；：、（）\(\)\[\]\"'“”‘’]", "", s)
        return s

    @classmethod
    def _norm_sent_start_map(cls, raw_text: str) -> List[int]:
        """Map each character index in `_normalize_text(raw_text)` to its sentence start index."""
        s1 = re.sub(r"[\s\u3000]+", " ", raw_text.lower().strip())
        break_positions = {m.end() for m in cls._SENT_SPLIT_RE.finditer(s1)}
        sent_starts: List[int] = []
        cur_start = 0
        norm_idx = 0
        for i, ch in enumerate(s1):
            if i in break_positions:
                cur_start = norm_idx
            if ch not in cls._STRIP_PUNCT_CHARS:
                sent_starts.append(cur_start)
                norm_idx += 1
        return sent_starts

    @classmethod
    def _has_standalone_status_sentence(cls, text_lower: str) -> bool:
        sents = [s for s in cls._SENT_SPLIT_RE.split(text_lower) if s.strip()]
        return any(cls._STANDALONE_STATUS_SENT_RE.match(s) for s in sents)

    @classmethod
    def _has_training_cert_only(cls, text_lower: str) -> bool:
        training_matches = list(cls._TRAINING_CERT_RE.finditer(text_lower))
        if not training_matches:
            return False
        training_spans = [m.span() for m in training_matches]
        for sm in cls._STATUTORY_CERT_RE.finditer(text_lower):
            if not any(t_lo <= sm.start() and sm.end() <= t_hi for t_lo, t_hi in training_spans):
                return False
        return True

    @classmethod
    def _subyear_re_for_mode(cls, check_subyear_tenure: Optional[bool]) -> Optional[re.Pattern]:
        if check_subyear_tenure is True:
            return cls._QA_EXP_SUB_YEAR_RE
        if check_subyear_tenure is False:
            return None
        return cls._SUB_YEAR_EXP_RE

    @classmethod
    def _has_non_negated_match(
        cls,
        text_lower: str,
        patterns: List[re.Pattern],
        tier: int,
        check_subyear_tenure: Optional[bool] = None,
    ) -> bool:
        sents = [s for s in cls._SENT_SPLIT_RE.split(text_lower) if s.strip()]
        if tier >= 2 and (
            any(cls._VERIF_FAIL_RE.search(s) for s in sents)
            or any(cls._STANDALONE_STATUS_SENT_RE.match(s) for s in sents)
        ):
            return False
        for sent in sents:
            if tier >= 2 and cls._has_disqualifying_subyear(sent, check_subyear_tenure=check_subyear_tenure):
                continue
            sub_clauses = [c for c in cls._COMMA_SPLIT_RE.split(sent) if c.strip()]
            for idx, clause in enumerate(sub_clauses):
                for pat in patterns:
                    for m in pat.finditer(clause):
                        pre_ctx = clause[: m.start()]
                        post_ctx = clause[m.end() :]
                        if tier == 1 and pat.pattern == r"\btrain(?:ing|ed)\b" and m.group(0) == "training":
                            if re.search(r"\b(?:after|following|prior\s+to)\s+(?:[a-z0-9'’-]+\s+){0,4}$", pre_ctx):
                                continue
                            if cls._QUALIFYING_MULTI_YEAR_RE.search(sent) and re.search(
                                r"(?:\(|with\s+|including\s+|\+\s*)\s*(?:[a-z0-9'’-]+\s+){0,4}$", pre_ctx
                            ):
                                continue
                        if cls._LOCAL_PRE_NEG_RE.search(pre_ctx) or cls._LOCAL_POST_NEG_RE.search(post_ctx):
                            continue
                        if tier >= 2 and idx > 0:
                            if any(
                                cls._PRE_AUTH_NEG_RE.search(prev_cl)
                                or (tier == 3 and cls._CERT_NEG_RE.search(prev_cl))
                                for prev_cl in sub_clauses[:idx]
                            ):
                                continue
                        if tier >= 2 and idx + 1 < len(sub_clauses):
                            if any(
                                cls._LOCAL_POST_NEG_RE.search(next_cl)
                                or cls._ANAPHORIC_AUTH_NEG_RE.search(next_cl)
                                or (tier == 3 and cls._CERT_NEG_RE.search(next_cl))
                                for next_cl in sub_clauses[idx + 1 :]
                            ):
                                continue
                        return True
        return False

    @classmethod
    def classify_verb_tier(cls, text: str, check_subyear_tenure: Optional[bool] = None) -> int:
        t = text.lower()
        has_training_cert_only = cls._has_training_cert_only(t)
        tier3_res = (
            cls._STRONG_CERT_ACTION_RES
            if (cls._CERT_NEG_RE.search(t) or has_training_cert_only or cls._has_standalone_status_sentence(t))
            else cls._STRONG_CERT_RES
        )
        if cls._has_non_negated_match(t, tier3_res, tier=3, check_subyear_tenure=check_subyear_tenure):
            return 3
        if cls._has_non_negated_match(t, cls._MODERATE_VERB_RES, tier=2, check_subyear_tenure=check_subyear_tenure):
            return 2
        if cls._has_non_negated_match(t, cls._WEAK_VERB_RES, tier=1, check_subyear_tenure=check_subyear_tenure):
            return 1
        return 0

    @classmethod
    def has_credential_proof(cls, text: str) -> bool:
        t = text.lower()
        if cls._CERT_NEG_RE.search(t) or cls._has_standalone_status_sentence(t):
            return False
        if cls._has_training_cert_only(t):
            return False
        has_cert = any(pat.search(t) for pat in cls._CERT_PROOF_RES)
        has_training_only = any(pat.search(t) for pat in cls._TRAINING_ONLY_RES)
        if has_training_only and not cls._CERT_EXPLICIT_RE.search(t):
            return False
        return has_cert

    def _extract_anchored_slice(self, resume_lines: List[str], anchor_lines: List[int]) -> Optional[str]:
        if not anchor_lines or not isinstance(anchor_lines, (list, tuple)):
            return None
        if len(anchor_lines) > 2 or not all(isinstance(x, int) and not isinstance(x, bool) for x in anchor_lines):
            return None
        start = anchor_lines[0]
        end = anchor_lines[-1]
        if start < 1 or end > len(resume_lines) or start > end:
            return None
        if (end - start + 1) > self.max_anchor_span:
            return None
        return " ".join(resume_lines[start - 1 : end])

    @staticmethod
    def _best_contiguous_sim(norm_ev: str, norm_src: str) -> Tuple[float, int, int]:
        if norm_ev in norm_src:
            idx = norm_src.find(norm_ev)
            return 1.0, idx, idx + len(norm_ev)
        m, n = len(norm_ev), len(norm_src)
        matcher = SequenceMatcher(None, norm_ev, norm_src)
        best = matcher.ratio()
        best_span = (0, n)
        if m < n:
            starts = [0] + [mt.end() for mt in re.finditer(r" ", norm_src)]
            if len(starts) < 4:
                starts = list(range(0, max(1, n - m + 1), max(1, m // 10)))
            for w in sorted({
                max(1, int(m * 0.78)),
                max(1, int(m * 0.88)),
                m,
                min(n, max(m + 1, int(m * 1.15))),
            }):
                if w >= n:
                    continue
                win_starts = sorted({max(0, min(i, n - w)) for i in starts})
                for i in win_starts:
                    matcher.set_seq2(norm_src[i : i + w])
                    if matcher.quick_ratio() > best:
                        r = matcher.ratio()
                        if r > best:
                            best = r
                            best_span = (i, i + w)
        return best, best_span[0], best_span[1]

    @classmethod
    def _has_disqualifying_subyear(cls, text_lower: str, check_subyear_tenure: Optional[bool] = None) -> bool:
        sub_re = cls._subyear_re_for_mode(check_subyear_tenure)
        if sub_re is None:
            return False
        for sent in cls._SENT_SPLIT_RE.split(text_lower):
            if sent.strip() and sub_re.search(sent) and not cls._QUALIFYING_MULTI_YEAR_RE.search(sent):
                return True
        return False

    @classmethod
    def _evidence_is_negated(cls, raw_ev: str, norm_ev: str, check_subyear_tenure: Optional[bool] = None) -> bool:
        ev_lower = raw_ev.lower()
        return bool(
            cls._EV_START_NEG_RE.match(norm_ev)
            or cls._INTERNAL_NEG_RE.search(ev_lower)
            or cls._CERT_NEG_RE.search(ev_lower)
            or cls._LOCAL_POST_NEG_RE.search(ev_lower)
            or cls._has_disqualifying_subyear(ev_lower, check_subyear_tenure=check_subyear_tenure)
        )

    @classmethod
    def _has_pre_or_internal_negation(
        cls,
        norm_ev: str,
        norm_src: str,
        lo: int,
        hi: int,
        sent_starts: Optional[List[int]] = None,
        check_subyear_tenure: Optional[bool] = None,
    ) -> bool:
        sent_lo = sent_starts[lo] if (sent_starts is not None and 0 <= lo < len(sent_starts)) else 0
        sent_hi = len(norm_src)
        if sent_starts is not None and 0 <= lo < len(sent_starts):
            for idx in range(lo + 1, len(sent_starts)):
                if sent_starts[idx] != sent_lo:
                    sent_hi = idx
                    break
        sent_text = norm_src[sent_lo:sent_hi]
        pre_window = norm_src[max(sent_lo, lo - 60) : lo]
        pre_sent = cls._SENT_SPLIT_RE.split(pre_window)[-1]
        if cls._SPAN_PRE_NEG_RE.search(pre_sent):
            return True
        sub_re = cls._subyear_re_for_mode(check_subyear_tenure)
        if sub_re is not None:
            if not cls._QUALIFYING_MULTI_YEAR_RE.search(norm_ev):
                for m_sub in sub_re.finditer(pre_sent):
                    if not cls._QUALIFYING_MULTI_YEAR_RE.search(pre_sent[m_sub.end() :]):
                        return True
        if (lo == 0 or lo == sent_lo) and cls._EV_START_NEG_RE.match(norm_src[lo:]):
            return True
        if sub_re is not None and (lo == 0 or lo == sent_lo) and sub_re.match(norm_src[lo:]) and not cls._QUALIFYING_MULTI_YEAR_RE.search(sent_text):
            return True
        src_negs = cls._INTERNAL_NEG_RE.findall(norm_src[lo:hi])
        ev_negs = cls._INTERNAL_NEG_RE.findall(norm_ev)
        if len(src_negs) > len(ev_negs):
            return True
        if (
            sub_re is not None
            and len(sub_re.findall(norm_src[lo:hi])) > len(sub_re.findall(norm_ev))
            and not cls._QUALIFYING_MULTI_YEAR_RE.search(sent_text)
        ):
            return True
        return False

    def check_span_alignment(
        self,
        resume_lines: List[str],
        evidence: str,
        anchor_lines: List[int],
        check_subyear_tenure: Optional[bool] = None,
    ) -> Tuple[bool, float]:
        if not evidence or not evidence.strip():
            return False, 0.0
        norm_ev = self._normalize_text(evidence)
        if not norm_ev:
            return False, 0.0

        anchored_text = self._extract_anchored_slice(resume_lines, anchor_lines)
        if anchored_text is None:
            return False, 0.0
        norm_src = self._normalize_text(anchored_text)
        if not norm_src:
            return False, 0.0
        sent_starts = self._norm_sent_start_map(anchored_text)

        # Never accept explicitly negated evidence for a verified span
        if self._evidence_is_negated(evidence, norm_ev, check_subyear_tenure=check_subyear_tenure):
            return False, 0.0

        if not self.use_fuzzy_span:
            if norm_ev not in norm_src:
                return False, 0.0
            for m in re.finditer(re.escape(norm_ev), norm_src):
                if not self._has_pre_or_internal_negation(
                    norm_ev, norm_src, m.start(), m.end(), sent_starts, check_subyear_tenure=check_subyear_tenure
                ):
                    return True, 1.0
            return False, 0.0

        # Ordered multi-segment contiguous verification BEFORE whole-string early return
        raw_segs = [s.strip() for s in self._MULTI_SEG_SPLIT_RE.split(evidence) if s.strip()]
        if len(raw_segs) >= 2:
            best_whole, _, _ = self._best_contiguous_sim(norm_ev, norm_src)
            cursor = 0
            seg_sims: List[Tuple[float, int]] = []
            total_len = 0
            for seg in raw_segs:
                nseg = self._normalize_text(seg)
                if len(nseg) < 6:
                    return False, min(best_whole, 0.0) if best_whole >= self.span_threshold else best_whole
                if self._evidence_is_negated(seg, nseg, check_subyear_tenure=check_subyear_tenure):
                    return False, 0.0
                s_sim, s_lo, s_hi = self._best_contiguous_sim(nseg, norm_src[cursor:])
                if s_sim < self.span_threshold:
                    return False, min(best_whole, s_sim)
                abs_lo = cursor + s_lo
                abs_hi = cursor + s_hi
                if self._has_pre_or_internal_negation(
                    nseg, norm_src, abs_lo, abs_hi, sent_starts, check_subyear_tenure=check_subyear_tenure
                ):
                    return False, 0.0
                seg_sims.append((s_sim, len(nseg)))
                total_len += len(nseg)
                cursor += s_hi
            weighted_sim = sum(s * l for s, l in seg_sims) / max(total_len, 1)
            return weighted_sim >= self.span_threshold, weighted_sim

        best_ratio, best_lo, best_hi = self._best_contiguous_sim(norm_ev, norm_src)
        if best_ratio >= self.span_threshold:
            if self._has_pre_or_internal_negation(
                norm_ev, norm_src, best_lo, best_hi, sent_starts, check_subyear_tenure=check_subyear_tenure
            ):
                return False, 0.0
            return True, best_ratio

        return False, best_ratio

    def _clause_matches_evidence(self, clause: str, norm_ev: str, norm_segs: List[str]) -> bool:
        ncl = self._normalize_text(clause)
        if not ncl:
            return False
        if (norm_ev and norm_ev in ncl) or any(ns in ncl for ns in norm_segs):
            return True
        if norm_ev and len(norm_ev) <= len(ncl) + 8:
            sim, _, _ = self._best_contiguous_sim(norm_ev, ncl)
            if sim >= self.span_threshold:
                return True
        return False

    def _extract_relevant_source_clauses(
        self,
        src_span: str,
        evidence: str,
        check_subyear_tenure: Optional[bool] = None,
    ) -> str:
        """Scope Rule 2 verb evaluation to matching sentence(s) and intra-sentence comma negation clauses."""
        ev_segs = [s.strip() for s in self._MULTI_SEG_SPLIT_RE.split(evidence) if s.strip()] or [evidence]
        norm_ev = self._normalize_text(evidence)
        norm_segs = [self._normalize_text(s) for s in ev_segs if len(self._normalize_text(s)) >= 4]

        sent_clauses = [c.strip() for c in self._SENT_SPLIT_RE.split(src_span) if c.strip()]
        if len(sent_clauses) > 1:
            pos_sents = [s for s in sent_clauses if self._clause_matches_evidence(s, norm_ev, norm_segs)]
            return "; ".join(pos_sents) if pos_sents else src_span

        clauses = [c.strip() for c in self._CLAUSE_SPLIT_RE.split(src_span) if c.strip()]
        if len(clauses) <= 1:
            return src_span

        matched: List[str] = []
        pos_matched = False
        sent_has_qual_yr = bool(self._QUALIFYING_MULTI_YEAR_RE.search(src_span.lower()))
        sub_re = self._subyear_re_for_mode(check_subyear_tenure)
        for cl in clauses:
            cl_lower = cl.lower()
            has_neg = bool(
                self._LOCAL_PRE_NEG_RE.search(cl_lower)
                or self._LOCAL_POST_NEG_RE.search(cl_lower)
                or self._ANAPHORIC_AUTH_NEG_RE.search(cl_lower)
                or self._CERT_NEG_RE.search(cl_lower)
                or (sub_re is not None and not sent_has_qual_yr and sub_re.search(cl_lower))
            )
            is_pos = self._clause_matches_evidence(cl, norm_ev, norm_segs)
            if is_pos and not has_neg:
                pos_matched = True
            if has_neg or is_pos:
                matched.append(cl)
        return ", ".join(matched) if (matched and pos_matched) else src_span

    _OVERLAP_STOPWORDS = frozenset({
        "and", "the", "for", "with", "from", "that", "this", "into", "over",
        "years", "year", "yrs", "months", "month", "not", "without", "formal",
    })

    def _has_overlapping_sibling_negation(
        self,
        src_span: str,
        evidence: str,
        check_subyear_tenure: Optional[bool] = None,
    ) -> bool:
        """Check whether a sibling sentence negates a capability overlapping `evidence`, asserts anaphoric authority denial, or contradicts matched evidence."""
        sent_clauses = [c.strip() for c in self._SENT_SPLIT_RE.split(src_span) if c.strip()]
        if len(sent_clauses) <= 1:
            return False
        ev_segs = [s.strip() for s in self._MULTI_SEG_SPLIT_RE.split(evidence) if s.strip()] or [evidence]
        norm_ev = self._normalize_text(evidence)
        norm_segs = [self._normalize_text(s) for s in ev_segs if len(self._normalize_text(s)) >= 4]
        matched_has_multi_year = bool(
            self._QUALIFYING_MULTI_YEAR_RE.search(evidence.lower())
            or any(
                self._QUALIFYING_MULTI_YEAR_RE.search(s.lower())
                for s in sent_clauses
                if self._clause_matches_evidence(s, norm_ev, norm_segs)
            )
        )
        ev_tokens = {
            t for t in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", evidence.lower())
            if len(t) >= 2 and t not in self._OVERLAP_STOPWORDS and not t.isdigit()
        }
        for sent in sent_clauses:
            sent_lower = sent.lower()
            if self._ANAPHORIC_AUTH_NEG_RE.search(sent_lower):
                return True
            if self._clause_matches_evidence(sent, norm_ev, norm_segs):
                if (
                    self._PRE_AUTH_NEG_RE.search(sent_lower)
                    or self._LOCAL_PRE_NEG_RE.search(sent_lower)
                    or self._LOCAL_POST_NEG_RE.search(sent_lower)
                    or self._VERIF_FAIL_RE.search(sent_lower)
                    or self._CERT_NEG_RE.search(sent_lower)
                    or self._INTERNAL_NEG_RE.search(sent_lower)
                    or self._has_disqualifying_subyear(sent_lower, check_subyear_tenure=check_subyear_tenure)
                ):
                    return True
                continue
            if not ev_tokens:
                continue
            has_relevant_neg = bool(
                self._PRE_AUTH_NEG_RE.search(sent_lower)
                or (
                    not self._CERT_NEG_RE.search(sent_lower)
                    and (
                        self._LOCAL_PRE_NEG_RE.search(sent_lower)
                        or self._LOCAL_POST_NEG_RE.search(sent_lower)
                        or (
                            self._has_disqualifying_subyear(sent_lower, check_subyear_tenure=check_subyear_tenure)
                            and not matched_has_multi_year
                        )
                    )
                )
            )
            if not has_relevant_neg:
                continue
            sent_tokens = {
                t for t in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", sent_lower)
                if len(t) >= 2 and t not in self._OVERLAP_STOPWORDS and not t.isdigit()
            }
            overlap = ev_tokens & sent_tokens
            if not overlap:
                continue
            if (
                self._WEAK_DISCLAIMER_RE.search(sent_lower)
                and not self._PRE_AUTH_NEG_RE.search(sent_lower)
                and len(overlap) < 2
                and overlap.issubset(self._GENERIC_TOPIC_NOUNS)
            ):
                continue
            return True
        return False

    def _matched_sentences_have_negation(
        self,
        src_span: str,
        evidence: str,
        check_subyear_tenure: Optional[bool] = None,
    ) -> bool:
        """Check explicit negation/invalidation only within sentence(s) actually matching evidence."""
        ev_segs = [s.strip() for s in self._MULTI_SEG_SPLIT_RE.split(evidence) if s.strip()] or [evidence]
        norm_ev = self._normalize_text(evidence)
        norm_segs = [self._normalize_text(s) for s in ev_segs if len(self._normalize_text(s)) >= 4]
        sent_clauses = [c.strip() for c in self._SENT_SPLIT_RE.split(src_span) if c.strip()]
        if len(sent_clauses) <= 1:
            targets = [src_span]
        else:
            matched_sents = [
                sent for sent in sent_clauses if self._clause_matches_evidence(sent, norm_ev, norm_segs)
            ]
            targets = matched_sents if matched_sents else [src_span]
        if any(
            self._evidence_is_negated(s, self._normalize_text(s), check_subyear_tenure=check_subyear_tenure)
            for s in targets
        ):
            return True
        sub_re = self._subyear_re_for_mode(check_subyear_tenure)
        if sub_re is not None:
            for s in targets:
                comma_clauses = [c.strip() for c in self._COMMA_SPLIT_RE.split(s) if c.strip()]
                if len(comma_clauses) <= 1:
                    continue
                for cl in comma_clauses:
                    cl_lower = cl.lower()
                    if (
                        self._clause_matches_evidence(cl, norm_ev, norm_segs)
                        and sub_re.search(cl_lower)
                        and not self._QUALIFYING_MULTI_YEAR_RE.search(cl_lower)
                        and not re.match(r"^(?:(?:with|including|plus)\b|含|包括)", cl_lower)
                    ):
                        return True
        return False

    def verify_card(self, resume_text: str, card: dict, role: str) -> Tuple[dict, List[str]]:
        """
        Deterministically verify every entry in the masked Evidence Card.
        Returns (verified_card, violation_codes).
        """
        resume_lines = resume_text.splitlines()
        req_map = {r.req_id: r for r in CompetencyOntology.get_requirements(role)}
        violations: List[str] = []
        verified_entries: List[dict] = []

        for entry in card["entries"]:
            e_copy = dict(entry)
            rid = e_copy["req_id"]
            req = req_map[rid]
            is_exp_tenure = rid == "QA_EXP"

            # Rule 4: Tier V must remain strictly blank
            if req.tier == "V":
                if (
                    e_copy.get("status") is not None
                    or str(e_copy.get("evidence") or "").strip()
                    or str(e_copy.get("claim_text") or "").strip()
                ):
                    violations.append(f"TACIT_EXTRAPOLATION_BLOCKED:{rid}")
                    e_copy["status"] = None
                    e_copy["evidence"] = ""
                    e_copy["anchor_lines"] = []
                    e_copy["claim_text"] = ""
                verified_entries.append(e_copy)
                continue

            status = e_copy["status"]
            evidence = str(e_copy.get("evidence") or "")
            claim_text = str(e_copy.get("claim_text") or evidence)
            raw_al = e_copy.get("anchor_lines")
            anchor_lines = list(raw_al) if isinstance(raw_al, (list, tuple)) else []

            if status == "verified":
                # Rule 1: Verbatim Span-Anchor Grounding Check
                aligned, sim_score = self.check_span_alignment(
                    resume_lines, evidence, anchor_lines, check_subyear_tenure=is_exp_tenure
                )
                e_copy["span_similarity"] = round(sim_score, 4)
                if not aligned:
                    violations.append(f"UNANCHORED_SPAN:{rid}")
                    e_copy["status"] = "conflicting"
                    e_copy["verifier_note"] = f"Span alignment failed (sim={sim_score:.2f} < {self.span_threshold})"
                    verified_entries.append(e_copy)
                    continue

                # Fetch actual source text at anchor for Rules 2 & 3, scoped to overlapping clause(s)
                src_span = self._extract_anchored_slice(resume_lines, anchor_lines) or ""
                eval_span = self._extract_relevant_source_clauses(
                    src_span, evidence, check_subyear_tenure=is_exp_tenure
                )
                notes: List[str] = []

                # Rule 2: Qualification-Verb Escalation Guard (Eq. 3)
                if self.use_verb_tier:
                    v_src = self.classify_verb_tier(eval_span, check_subyear_tenure=is_exp_tenure)
                    v_claim_raw = self.classify_verb_tier(claim_text, check_subyear_tenure=is_exp_tenure)
                    v_req_floor = req.min_verb_tier if (req.min_verb_tier >= 3 or req.tier == "P" or v_src > 0) else 0
                    v_max = max(v_req_floor, v_claim_raw)
                    if v_src < v_max:
                        if req.min_verb_tier >= 3 and v_src < 3 and v_claim_raw <= v_src:
                            violations.append(f"VERB_ESCALATION:{rid}:src_v{v_src}_req_v{req.min_verb_tier}")
                            notes.append(
                                f"Weak source verb tier ({v_src}) cannot verify Tier-{req.min_verb_tier} statutory requirement"
                            )
                        else:
                            violations.append(f"VERB_ESCALATION:{rid}:src_v{v_src}_max_v{v_max}")
                            notes.append(f"Claim/requirement verb tier ({v_max}) exceeds source span tier ({v_src})")
                    elif (
                        (v_src == 0 and self._matched_sentences_have_negation(
                            src_span, evidence, check_subyear_tenure=is_exp_tenure
                        ))
                        or self._VERIF_FAIL_RE.search(src_span.lower())
                        or self._has_standalone_status_sentence(src_span.lower())
                        or (req.requires_cert_proof and self._CERT_NEG_RE.search(src_span.lower()))
                        or self._has_overlapping_sibling_negation(
                            src_span, evidence, check_subyear_tenure=is_exp_tenure
                        )
                    ):
                        violations.append(f"VERB_ESCALATION:{rid}:negated_source_span")
                        notes.append("Anchored source span contains explicit negation/invalidation")

                # Rule 3: Mandatory Credential Proof Rule (Eq. 3, evaluated on full anchored src_span)
                if self.use_cert_proof and req.requires_cert_proof and not self.has_credential_proof(src_span):
                    violations.append(f"MISSING_CREDENTIAL_PROOF:{rid}")
                    notes.append("Statutory certificate requirement lacks verifiable credential proof")

                if notes:
                    e_copy["status"] = "conflicting"
                    e_copy["verifier_note"] = "; ".join(notes)

            verified_entries.append(e_copy)

        verified_card = {
            "candidate_id": card["candidate_id"],
            "role": role,
            "entries": verified_entries,
            "verifier_violations": violations,
        }
        return verified_card, violations


# =============================================================================
# 4. Finite-State Decision Gate (Eq. 5 & Eq. 6)
# =============================================================================

@dataclass
class GateTransitionResult:
    next_state: str  # "s_term", "s_esc", "s_block"
    final_decision: Optional[str]  # "0", "1", "Esc", or None (blocked)
    error_code: Optional[str] = None
    audit_entry: Dict[str, object] = field(default_factory=dict)


class FSMDecisionGate:
    """
    Finite-state human decision gate enforcing:
      - G_M: Mandatory criteria guard on admission (D=1)
      - G_esc: Hard escalation lock whenever conflicting/verifier flags exist
      - G_rsn: Minimum justification length |rsn| >= L_min on Adjust
    """

    def __init__(self, l_min: int = 15, enforce_hard_escalation_lock: bool = True):
        self.l_min = l_min
        self.enforce_hard_escalation_lock = enforce_hard_escalation_lock

    @staticmethod
    def _mandatory_guard_holds(entries: List[dict]) -> bool:
        m_entries = [e for e in entries if e.get("tier") == "M"]
        return bool(m_entries) and all(e.get("status") == "verified" for e in m_entries)

    @staticmethod
    def _has_unresolved_conflict(card: dict) -> bool:
        any_conflict = any(e.get("status") == "conflicting" for e in card["entries"] if e.get("tier") in ("M", "P"))
        any_ver_flag = bool(card.get("verifier_violations"))
        return any_conflict or any_ver_flag

    def step(
        self,
        card: dict,
        action: str,  # "Confirm", "Adjust", "Escalate"
        proposed_decision: Optional[int] = None,  # 0 or 1
        reason: str = "",
        adjusted_statuses: Optional[Dict[str, str]] = None,
    ) -> GateTransitionResult:
        entries = [dict(e) for e in card["entries"]]
        ts = time.time()

        if action == "Escalate":
            return GateTransitionResult(
                next_state="s_esc",
                final_decision="Esc",
                audit_entry={"ts": ts, "action": "Escalate", "reason": reason, "violations": card.get("verifier_violations", [])},
            )

        if action in ("Confirm", "Adjust"):
            if type(proposed_decision) is not int or proposed_decision not in (0, 1):
                return GateTransitionResult(
                    next_state="s_block",
                    final_decision=None,
                    error_code="ERR_INVALID_DECISION",
                    audit_entry={"ts": ts, "action": action, "proposed_decision": proposed_decision, "blocked": True},
                )

        if action == "Confirm":
            if proposed_decision == 1:
                if self.enforce_hard_escalation_lock and self._has_unresolved_conflict(card):
                    return GateTransitionResult(
                        next_state="s_block",
                        final_decision=None,
                        error_code="ERR_UNRESOLVED_CONFLICT_LOCK",
                        audit_entry={"ts": ts, "action": "Confirm(1)", "blocked": True},
                    )
                if not self._mandatory_guard_holds(entries):
                    return GateTransitionResult(
                        next_state="s_block",
                        final_decision=None,
                        error_code="ERR_MANDATORY_GUARD_FAILED",
                        audit_entry={"ts": ts, "action": "Confirm(1)", "blocked": True},
                    )
            return GateTransitionResult(
                next_state="s_term",
                final_decision=str(proposed_decision),
                audit_entry={"ts": ts, "action": f"Confirm({proposed_decision})", "reason": reason},
            )

        if action == "Adjust":
            if len(reason.strip()) < self.l_min:
                return GateTransitionResult(
                    next_state="s_block",
                    final_decision=None,
                    error_code="ERR_JUSTIFICATION_TOO_SHORT",
                    audit_entry={"ts": ts, "action": "Adjust", "blocked": True},
                )
            if self.enforce_hard_escalation_lock and proposed_decision == 1 and self._has_unresolved_conflict(card):
                return GateTransitionResult(
                    next_state="s_block",
                    final_decision=None,
                    error_code="ERR_UNRESOLVED_CONFLICT_LOCK",
                    audit_entry={"ts": ts, "action": "Adjust(1)", "blocked": True},
                )
            if adjusted_statuses:
                for rid, new_status in adjusted_statuses.items():
                    if new_status not in CLOSED_STATUS_ENUM:
                        return GateTransitionResult(
                            next_state="s_block",
                            final_decision=None,
                            error_code="ERR_INVALID_ENUM_STATUS",
                            audit_entry={"ts": ts, "action": "Adjust", "req_id": rid, "status": new_status, "blocked": True},
                        )
                for e in entries:
                    if e["req_id"] in adjusted_statuses and e["tier"] in ("M", "P"):
                        e["status"] = adjusted_statuses[e["req_id"]]
            adjusted_card = {**card, "entries": entries}
            if self.enforce_hard_escalation_lock and proposed_decision == 1 and self._has_unresolved_conflict(adjusted_card):
                return GateTransitionResult(
                    next_state="s_block",
                    final_decision=None,
                    error_code="ERR_UNRESOLVED_CONFLICT_LOCK",
                    audit_entry={"ts": ts, "action": "Adjust(1)", "blocked": True},
                )
            if proposed_decision == 1 and not self._mandatory_guard_holds(entries):
                return GateTransitionResult(
                    next_state="s_block",
                    final_decision=None,
                    error_code="ERR_MANDATORY_GUARD_FAILED",
                    audit_entry={"ts": ts, "action": "Adjust(1)", "blocked": True},
                )
            return GateTransitionResult(
                next_state="s_term",
                final_decision=str(proposed_decision),
                audit_entry={"ts": ts, "action": f"Adjust({proposed_decision})", "reason": reason},
            )

        return GateTransitionResult(
            next_state="s_block",
            final_decision=None,
            error_code="ERR_INVALID_ACTION",
            audit_entry={"ts": ts, "action": action, "blocked": True},
        )


# =============================================================================
# 5. Reference Prompt Assembler & OpenAI-Compatible vLLM Extractor (Algorithm 1)
# =============================================================================

REFERENCE_SYSTEM_PROMPT = (
    "You are a constrained compliance evidence extractor for medical device talent screening "
    "under ISO 13485:2016 (China GB/T 42061-2022), NMPA GMP, and China State Council Order No. 739/797.\n"
    "RULES:\n"
    "1. Extract ONLY verbatim resume quotes and exact 1-based line anchors for Tier-M and Tier-P "
    "requirements listed in the JSON Schema.\n"
    "2. NEVER output scalar fit scores, rankings, or hire/reject verdicts.\n"
    "3. NEVER infer or emit submerged tacit traits (Tier-V slots are masked from the schema).\n"
    "4. Status must belong strictly to {verified, not_provided, conflicting}.\n"
    "5. Do NOT inflate weak participation verbs into certified or leadership claims.\n"
    "6. Ground cross-standard equivalence strictly in the Top-3 version-pinned regulatory clauses provided."
)


def build_extraction_prompt(
    candidate_id: str,
    role: str,
    resume_lines: List[str],
    rag_clauses: List[Tuple[RegulatoryClause, float]],
) -> Dict[str, Any]:
    """
    Assemble the Algorithm 1 Stage-2 prompt payload and compiled JSON Schema.

    Note: Uses the open English reference prompt template documented in
    `prompts/eica_extraction_reference_prompt.md` (the partner firm's internal
    Chinese production prompt text is withheld under SOP/PIPL confidentiality).
    """
    schema = OntologySchemaMasker.compile_generation_schema(role)
    extractable = [r for r in CompetencyOntology.get_requirements(role) if r.tier in ("M", "P")]

    clause_lines = [
        f"- [{doc.doc_id} | {doc.version_id} | score={score:.3f}] {doc.title}: {doc.text}"
        for doc, score in rag_clauses
    ]
    req_lines = [
        f"- {r.req_id} (Tier {r.tier}, min_verb_tier={r.min_verb_tier}, cert_proof={int(r.requires_cert_proof)}): {r.label}"
        for r in extractable
    ]
    dossier_lines = [f"L{idx}: {line}" for idx, line in enumerate(resume_lines, start=1)]

    user_prompt = (
        f"Target Role: {role}\n"
        f"Candidate ID: {candidate_id}\n\n"
        f"### Version-Pinned Regulatory Clauses (Top-{len(rag_clauses)} Active D_pin)\n"
        + ("\n".join(clause_lines) if clause_lines else "(none)")
        + "\n\n### Extractable Competency Requirements (Tier M & Tier P Only; Tier V Masked)\n"
        + "\n".join(req_lines)
        + "\n\n### Numbered De-Identified Candidate Dossier\n"
        + "\n".join(dossier_lines)
        + "\n\nReturn a single JSON object conforming strictly to the compiled schema."
    )
    return {
        "messages": [
            {"role": "system", "content": REFERENCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "json_schema": schema,
    }


class VLLMExtractor:
    """
    OpenAI-compatible vLLM client wrapper implementing Algorithm 1 `LLMExtract`
    (lines 3-4: greedy decoding T=0, seed=42, retry <= 2, followed by host-side
    `OntologySchemaMasker.enforce_mask_and_validate`).

    Supports dependency-injected `transport` callable for deterministic offline tests.
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8000/v1/chat/completions",
        model: str = "Qwen/Qwen2.5-32B-Instruct",
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 2048,
        max_retries: int = 2,
        timeout_sec: float = 60.0,
        transport: Optional[Callable[[dict], str]] = None,
    ):
        self.endpoint = endpoint
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout_sec = timeout_sec
        self._transport = transport

    def _default_http_transport(self, payload: dict) -> str:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
        return parsed["choices"][0]["message"]["content"]

    def extract(
        self,
        candidate_id: str,
        role: str,
        resume_lines: List[str],
        rag_clauses: List[Tuple[RegulatoryClause, float]],
    ) -> Tuple[dict, List[str]]:
        """Execute LLMExtract with retry <= max_retries and host-side schema hard masking."""
        assembled = build_extraction_prompt(candidate_id, role, resume_lines, rag_clauses)
        payload = {
            "model": self.model,
            "messages": assembled["messages"],
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "eica_evidence_card",
                    "strict": True,
                    "schema": assembled["json_schema"],
                },
            },
        }

        transport = self._transport or self._default_http_transport
        last_err: Optional[Exception] = None
        for _attempt in range(self.max_retries + 1):
            try:
                raw_text = transport(payload)
                raw_card = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
                if not isinstance(raw_card, dict):
                    raise ValueError("Extractor output must be a JSON object")
                raw_card.setdefault("candidate_id", candidate_id)
                raw_card.setdefault("role", role)
                return OntologySchemaMasker.enforce_mask_and_validate(raw_card, role)
            except Exception as exc:
                last_err = exc

        raise RuntimeError(
            f"LLMExtract failed after {self.max_retries + 1} attempts: {last_err}"
        )
