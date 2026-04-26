from __future__ import annotations

from .types import Example

EXAMPLES: list[Example] = [
    Example(
        sentence="The trophy did not fit in the suitcase because it was large.",
        subject="trophy",
        object="suitcase",
        adjective="large",
        expected="subject",
        scenario="lexical_core",
    ),
    Example(
        sentence="The trophy did not fit in the suitcase because it was small.",
        subject="trophy",
        object="suitcase",
        adjective="small",
        expected="object",
        scenario="lexical_core",
    ),
    Example(
        sentence="The statue did not fit in the cabinet because it was huge.",
        subject="statue",
        object="cabinet",
        adjective="huge",
        expected="subject",
        scenario="synonym_generalization",
    ),
    Example(
        sentence="The box did not fit in the drawer because it was tiny.",
        subject="box",
        object="drawer",
        adjective="tiny",
        expected="object",
        scenario="synonym_generalization",
    ),
    Example(
        sentence="The robot did not fit in the container because it was oversized.",
        subject="robot",
        object="container",
        adjective="oversized",
        expected="subject",
        scenario="conflicting_evidence",
    ),
    Example(
        sentence="The statue did not fit in the suitcase because it was narrow.",
        subject="statue",
        object="suitcase",
        adjective="narrow",
        expected="object",
        scenario="synonym_generalization",
    ),
]

HELDOUT_EXAMPLES: list[Example] = [
    Example(
        sentence="The sculpture did not fit in the case because it was enormous.",
        subject="sculpture",
        object="case",
        adjective="enormous",
        expected="subject",
        scenario="synonym_generalization",
    ),
    Example(
        sentence="The package did not fit in the slot because it was cramped.",
        subject="package",
        object="slot",
        adjective="cramped",
        expected="object",
        scenario="synonym_generalization",
    ),
]

DYNAMIC_EXAMPLES: list[Example] = [
    Example(
        sentence="The trophy did not fit in the suitcase because it was large.",
        subject="trophy",
        object="suitcase",
        adjective="large",
        expected="subject",
        scenario="lexical_core",
    ),
    Example(
        sentence="The trophy did not fit in the suitcase because it was large. Wait, the trophy is made of shrinking foam, so 'large' means it shrank drastically.",
        subject="trophy",
        object="suitcase",
        adjective="large",
        expected="object",
        scenario="conflicting_evidence",
        context_rules=[
            # Context explicitly overrides the standard semantic embedding
            "<large --> small_like>. %1.00;0.95%"
        ]
    ),
]

_NOUN_PAIRS: list[tuple[str, str]] = [
    ("trophy", "suitcase"),
    ("statue", "cabinet"),
    ("robot", "container"),
    ("table", "closet"),
    ("package", "slot"),
    ("drum", "locker"),
    ("vase", "bag"),
    ("monitor", "crate"),
]


def _fit_sentence(subject: str, object_: str, adjective: str) -> str:
    return f"The {subject} did not fit in the {object_} because it was {adjective}."


def build_benchmark_examples() -> list[Example]:
    """
    Build a larger benchmark with scenario tags for ablation reporting.

    Scenarios:
      lexical_core: exact adjective words 'large'/'small'
      synonym_generalization: held-out synonyms
      conflicting_evidence: context rules invert the initial semantic cue
      multihop_chain: adjectives map to bulky/compact concepts requiring multi-hop rules
    """
    rows: list[Example] = []

    for subject, object_ in _NOUN_PAIRS:
        rows.append(
            Example(
                sentence=_fit_sentence(subject, object_, "large"),
                subject=subject,
                object=object_,
                adjective="large",
                expected="subject",
                scenario="lexical_core",
            )
        )
        rows.append(
            Example(
                sentence=_fit_sentence(subject, object_, "small"),
                subject=subject,
                object=object_,
                adjective="small",
                expected="object",
                scenario="lexical_core",
            )
        )

    synonym_adjectives = [
        ("huge", "subject"),
        ("enormous", "subject"),
        ("tiny", "object"),
        ("cramped", "object"),
        ("narrow", "object"),
    ]
    for idx, (subject, object_) in enumerate(_NOUN_PAIRS):
        adj, expected = synonym_adjectives[idx % len(synonym_adjectives)]
        rows.append(
            Example(
                sentence=_fit_sentence(subject, object_, adj),
                subject=subject,
                object=object_,
                adjective=adj,
                expected=expected,
                scenario="synonym_generalization",
            )
        )

    for subject, object_ in _NOUN_PAIRS:
        rows.append(
            Example(
                sentence=_fit_sentence(subject, object_, "large"),
                subject=subject,
                object=object_,
                adjective="large",
                expected="object",
                scenario="conflicting_evidence",
                context_rules=["<large --> small_like>. %1.00;0.95%"],
            )
        )
        rows.append(
            Example(
                sentence=_fit_sentence(subject, object_, "small"),
                subject=subject,
                object=object_,
                adjective="small",
                expected="subject",
                scenario="conflicting_evidence",
                context_rules=["<small --> large_like>. %1.00;0.95%"],
            )
        )

    multihop_adjectives = [
        ("bulky", "subject"),
        ("massive", "subject"),
        ("compact", "object"),
        ("slender", "object"),
    ]
    for idx, (subject, object_) in enumerate(_NOUN_PAIRS):
        adj, expected = multihop_adjectives[idx % len(multihop_adjectives)]
        rows.append(
            Example(
                sentence=_fit_sentence(subject, object_, adj),
                subject=subject,
                object=object_,
                adjective=adj,
                expected=expected,
                scenario="multihop_chain",
            )
        )

    return rows


_SUBJECT_BASES = [
    "trophy",
    "statue",
    "robot",
    "table",
    "package",
    "drum",
    "vase",
    "monitor",
    "lamp",
    "printer",
    "helmet",
    "guitar",
]

_OBJECT_BASES = [
    "suitcase",
    "cabinet",
    "container",
    "closet",
    "slot",
    "locker",
    "bag",
    "crate",
    "drawer",
    "box",
    "case",
    "tube",
]


def build_research_examples(n_pairs: int = 40) -> list[Example]:
    """
    Construct a larger deterministic benchmark for statistical evaluation.

    Each noun pair contributes six examples:
      - lexical_core: large/small
      - synonym_generalization: one held-out synonym
      - conflicting_evidence: two context-overridden cases
      - multihop_chain: one bulky/compact concept case
    """
    if n_pairs <= 0:
        raise ValueError("n_pairs must be > 0")

    rows: list[Example] = []
    synonym_cycle = [
        ("huge", "subject"),
        ("enormous", "subject"),
        ("tiny", "object"),
        ("cramped", "object"),
        ("narrow", "object"),
    ]
    multihop_cycle = [
        ("bulky", "subject"),
        ("massive", "subject"),
        ("compact", "object"),
        ("slender", "object"),
    ]

    for idx in range(n_pairs):
        subject = f"{_SUBJECT_BASES[idx % len(_SUBJECT_BASES)]}_{idx}"
        object_ = f"{_OBJECT_BASES[idx % len(_OBJECT_BASES)]}_{idx}"

        rows.append(
            Example(
                sentence=_fit_sentence(subject, object_, "large"),
                subject=subject,
                object=object_,
                adjective="large",
                expected="subject",
                scenario="lexical_core",
            )
        )
        rows.append(
            Example(
                sentence=_fit_sentence(subject, object_, "small"),
                subject=subject,
                object=object_,
                adjective="small",
                expected="object",
                scenario="lexical_core",
            )
        )

        syn_adj, syn_expected = synonym_cycle[idx % len(synonym_cycle)]
        rows.append(
            Example(
                sentence=_fit_sentence(subject, object_, syn_adj),
                subject=subject,
                object=object_,
                adjective=syn_adj,
                expected=syn_expected,
                scenario="synonym_generalization",
            )
        )

        rows.append(
            Example(
                sentence=_fit_sentence(subject, object_, "large"),
                subject=subject,
                object=object_,
                adjective="large",
                expected="object",
                scenario="conflicting_evidence",
                context_rules=["<large --> small_like>. %1.00;0.95%"],
            )
        )
        rows.append(
            Example(
                sentence=_fit_sentence(subject, object_, "small"),
                subject=subject,
                object=object_,
                adjective="small",
                expected="subject",
                scenario="conflicting_evidence",
                context_rules=["<small --> large_like>. %1.00;0.95%"],
            )
        )

        mh_adj, mh_expected = multihop_cycle[idx % len(multihop_cycle)]
        rows.append(
            Example(
                sentence=_fit_sentence(subject, object_, mh_adj),
                subject=subject,
                object=object_,
                adjective=mh_adj,
                expected=mh_expected,
                scenario="multihop_chain",
            )
        )

    return rows
