from __future__ import annotations

from ragnav.answering.inline_citations import validate_inline_citations


def main() -> None:
    allowed = {"doc#b0", "doc#b1"}

    ok = validate_inline_citations("Claim one [[doc#b0]]. Claim two [[doc#b1]]!", allowed_block_ids=allowed)
    assert ok["ok"] is True

    bad = validate_inline_citations("No citations here.", allowed_block_ids=allowed)
    assert bad["ok"] is False

    unknown = validate_inline_citations("Bad cite. [[doc#b999]]", allowed_block_ids=allowed)
    assert unknown["ok"] is False

    print("citation_eval_ok")


if __name__ == "__main__":
    main()

