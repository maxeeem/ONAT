from __future__ import annotations

from .types import Example

EXAMPLES: list[Example] = [
    Example(
        sentence="The trophy did not fit in the suitcase because it was large.",
        subject="trophy",
        object="suitcase",
        adjective="large",
        expected="subject",
    ),
    Example(
        sentence="The trophy did not fit in the suitcase because it was small.",
        subject="trophy",
        object="suitcase",
        adjective="small",
        expected="object",
    ),
    Example(
        sentence="The statue did not fit in the cabinet because it was huge.",
        subject="statue",
        object="cabinet",
        adjective="huge",
        expected="subject",
    ),
    Example(
        sentence="The box did not fit in the drawer because it was tiny.",
        subject="box",
        object="drawer",
        adjective="tiny",
        expected="object",
    ),
    Example(
        sentence="The robot did not fit in the container because it was oversized.",
        subject="robot",
        object="container",
        adjective="oversized",
        expected="subject",
    ),
    Example(
        sentence="The statue did not fit in the suitcase because it was narrow.",
        subject="statue",
        object="suitcase",
        adjective="narrow",
        expected="object",
    ),
]

HELDOUT_EXAMPLES: list[Example] = [
    Example(
        sentence="The sculpture did not fit in the case because it was enormous.",
        subject="sculpture",
        object="case",
        adjective="enormous",
        expected="subject",
    ),
    Example(
        sentence="The package did not fit in the slot because it was cramped.",
        subject="package",
        object="slot",
        adjective="cramped",
        expected="object",
    ),
]

DYNAMIC_EXAMPLES: list[Example] = [
    Example(
        sentence="The trophy did not fit in the suitcase because it was large.",
        subject="trophy",
        object="suitcase",
        adjective="large",
        expected="subject",
    ),
    Example(
        sentence="The trophy did not fit in the suitcase because it was large. Wait, the trophy is made of shrinking foam, so 'large' means it shrank drastically.",
        subject="trophy",
        object="suitcase",
        adjective="large",
        expected="object",
        context_rules=[
            # Context explicitly overrides the standard semantic embedding
            "<large --> small_like>. %1.00;0.95%"
        ]
    ),
]
