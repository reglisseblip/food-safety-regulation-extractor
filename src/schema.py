"""
Regulatory matrix structure (Pydantic v2 schema) — simple but complete.

A `Rule` = one matrix row = one regulation.
Two families: quantitative limit (MAX / MIN / RANGE) · prohibition/permission.
Shared core + ONE subject of your choice: `substance` / `group` / `labeling` / `claim`.

Official identifier (INS, E-number…): ONE optional field, kept as written when present.
No lookup / derivation for now (see doc/probleme.md P1: only useful to link substances
across countries — we'll deal with it then).

Deliberate tolerance: validation enforces the single subject, but NOT rule_type ↔ value
coherence (a MAX with no value_max passes). Requiring the value would lose the whole rule when
the model misreads ONE field; we'd rather keep the row and flag it (quality.needs_review) —
"flagged, never dropped".
"""
from __future__ import annotations

from enum import Enum
from typing import ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ============================ closed lists (for filtering) ============================
class SubstanceClass(str, Enum):
    additive = "additive"
    contaminant = "contaminant"
    nutrient = "nutrient"
    enzyme = "enzyme"
    flavouring = "flavouring"
    allergen = "allergen"
    residue = "residue"            # pesticide / veterinary residue
    microorganism = "microorganism"
    other = "other"


class Theme(str, Enum):
    food_additives = "food_additives"
    contaminants = "contaminants"
    residues = "residues"
    microbiology = "microbiology"
    labeling = "labeling"
    claims = "claims"
    fortification = "fortification"
    other = "other"


class ConsumerGroup(str, Enum):
    general = "general"
    infants = "infants"
    young_children = "young_children"
    medical_fsmp = "medical_fsmp"
    coeliac = "coeliac"
    other = "other"


class RuleType(str, Enum):
    MAX = "MAX"                          # maximum limit
    MIN = "MIN"                          # minimum limit
    RANGE = "RANGE"                      # min-max range
    PROHIBITED = "PROHIBITED"            # prohibition
    PERMITTED = "PERMITTED"              # permission


class Basis(str, Enum):
    as_consumed = "as_consumed"
    as_sold = "as_sold"
    per_100g = "per_100g"
    per_100ml = "per_100ml"
    per_100kcal = "per_100kcal"
    per_serving = "per_serving"
    per_day = "per_day"
    dry_matter = "dry_matter"
    on_fat_basis = "on_fat_basis"
    other = "other"


class ConditionType(str, Enum):
    except_ = "except"
    only_if = "only_if"
    when = "when"
    note_ref = "note_ref"


class LabelKind(str, Enum):
    mandatory_element = "mandatory_element"
    declaration = "declaration"
    format = "format"
    naming = "naming"
    exemption = "exemption"


# ============================ blocks ============================
class _Base(BaseModel):
    """Shared base: no unexpected field accepted — catches an LLM hallucination
    in a rule as well as in a sub-model."""
    model_config = ConfigDict(extra="forbid")


class Substance(_Base):
    """'Chemical' subject: additive, contaminant, nutrient, residue…"""
    name: str = Field(..., description="Nom en anglais (normalisé ; l'original reste dans source_snippet)")
    substance_class: SubstanceClass
    identifier: Optional[str] = Field(None, description="Numéro officiel tel qu'écrit (INS, n° E…), gardé seulement s'il est présent")
    chemical_form: Optional[str] = Field(None, description="Forme/spéciation, ex. 'arsenic inorganique'")


class GroupLimit(_Base):
    """Limit SHARED across several substances ("singly or in combination"): the value targets the SUM."""
    members: list[str] = Field(..., description="Substances du groupe, ex. ['E 471','E 472a']")


class Labeling(_Base):
    """Labeling rule: no substance, no numeric value.
    (A labeling trigger / condition goes in `conditions`, as everywhere else.)"""
    kind: LabelKind
    element: str = Field(..., description="Élément concerné, ex. 'liste des ingrédients'")


class Condition(_Base):
    """A restriction / note WITHOUT a different value.
    (If the exception CHANGES the value for a sub-case → a separate RULE, see doc/probleme.md P9.)"""
    type: ConditionType
    text: str = Field(..., description="La condition en clair")
    ref: Optional[str] = Field(None, description="N° de note référencé, ex. '468'")


class Norm(_Base):
    """The required value (when numeric)."""
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    unit: Optional[str] = Field(None, description="ex. mg/kg, mg/l, %, Bq/kg, UI")
    basis: Optional[Basis] = None


class Provenance(_Base):
    """Where it comes from — to verify and cite."""
    document_ref: str = Field(..., description="Document source (obligatoire)")
    source_snippet: str = Field(..., description="Extrait EXACT (obligatoire)")
    page: Optional[int] = None
    version: Optional[str] = Field(None, description="Année / marqueur d'amendement")

    # local pointers into the source: jurisdiction-specific, often absent, never universal —
    # hence their place here (provenance) and not in the core.
    legal_reference: Optional[str] = Field(None, description="Table / annexe / article, ex. 'Table 2'")
    category_code: Optional[str] = Field(None, description="Code catégorie tel qu'écrit, ex. '05.0' (GSFA) — hiérarchique là où il existe ; servira à l'héritage local (P4)")


class Quality(_Base):
    """Extraction is always done by an LLM."""
    confidence: Optional[float] = Field(None, ge=0, le=1)
    needs_review: bool = Field(False, description="Cas douteux : signalé, jamais supprimé")


# ============================ the rule (one matrix row) ============================
class Rule(_Base):
    """One rule = one row. Core + ONE subject (substance / group / labeling / claim)."""

    # the 4 subjects, listed ONCE — used by _coherence, subject and subject_label
    _SUBJECTS: ClassVar[tuple[str, ...]] = ("substance", "group", "labeling", "claim")

    # core
    jurisdiction: str = Field(..., description="Pays / juridiction, ex. 'Codex', 'EU', 'Thailand'")
    theme: Theme
    rule_type: RuleType
    product_category: Optional[str] = Field(None, description="Catégorie d'aliment (libellé, ex. 'Confectionery') ; le code catégorie éventuel va dans provenance.category_code")
    consumer_group: ConsumerGroup = ConsumerGroup.general

    # subject (one of the four)
    substance: Optional[Substance] = None
    group: Optional[GroupLimit] = None
    labeling: Optional[Labeling] = None
    claim: Optional[str] = Field(None, description="Texte de l'allégation, ex. 'sans gluten'")

    # value, conditions, source
    norm: Norm = Field(default_factory=Norm)
    conditions: list[Condition] = Field(default_factory=list)
    provenance: Provenance
    quality: Quality = Field(default_factory=Quality)

    # The LLM often writes `null` where it has nothing to say (a rule with no numeric value → norm=null,
    # no conditions → conditions=null). A defaulted field still rejects `null` (the default applies ONLY
    # when the key is absent). Without this tolerance, ONE `null` rule made the WHOLE batch (RuleBatch)
    # fail and the entire chunk was lost -> unstable recall. We treat `null` as "nothing said" = the default.
    @field_validator("norm", "quality", "conditions", "consumer_group", "theme", "product_category",
                     mode="before")
    @classmethod
    def _null_vaut_defaut(cls, v, info):
        if v is not None:
            return v
        return {"norm": {}, "quality": {}, "conditions": [], "consumer_group": "general",
                "theme": "other", "product_category": None}[info.field_name]

    @model_validator(mode="after")
    def _coherence(self):
        present = [n for n in self._SUBJECTS if getattr(self, n)]
        if not present:
            raise ValueError("Règle sans sujet — exactement UN parmi "
                             "substance / group / labeling / claim.")
        if len(present) > 1:
            raise ValueError("Règle à PLUSIEURS sujets — exactement UN parmi "
                             "substance / group / labeling / claim (les autres à null).")
        return self

    # ---- the subject, accessible in one step (the 4-way branching lives only here) ----
    @property
    def subject(self) -> Optional[tuple]:
        """The rule's subject as (type, content) — e.g. ("substance", <Substance>);
        None if no subject (rejected at validation). Everyone (dedup, matrix, benchmark) goes
        through here: no more copies of the branching drifting across several files."""
        for name in self._SUBJECTS:
            value = getattr(self, name)
            if value:
                return name, value
        return None

    def subject_label(self) -> Optional[str]:
        """The subject's name as a single string — the SAME everywhere (dedup, matrix, benchmark).
        Derived from `subject`: reading the 4 fields lives in a single place."""
        s = self.subject
        if s is None:
            return None
        kind, value = s
        if kind == "substance":
            return value.name
        if kind == "group":
            return "group:" + "+".join(value.members)
        if kind == "labeling":
            return value.element
        return value  # claim: already a string
