from __future__ import annotations

from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from typing import Any, Optional

from ..graph import BlockGraph, Edge
from ..models import Block, Document


@dataclass(frozen=True)
class EmailIngestOptions:
    doc_id_prefix: str = "email:"
    max_body_chars: int = 8000


def ingest_eml_bytes_graph(
    data: bytes,
    *,
    name: str = "message.eml",
    metadata: Optional[dict[str, Any]] = None,
    opts: Optional[EmailIngestOptions] = None,
) -> BlockGraph:
    """
    RFC822 .eml → BlockGraph
    - email header block
    - email body block
    - reply_to edge via In-Reply-To / References (best-effort)
    """
    opts = opts or EmailIngestOptions()
    msg = BytesParser(policy=policy.default).parsebytes(data)

    subj = str(msg.get("Subject", "") or "")
    msg_id = str(msg.get("Message-Id", "") or "").strip()
    in_reply_to = str(msg.get("In-Reply-To", "") or "").strip()
    refs = str(msg.get("References", "") or "").strip()

    doc_meta: dict[str, Any] = {"type": "email", "subject": subj}
    if msg_id:
        doc_meta["message_id"] = msg_id
    if metadata:
        doc_meta.update(metadata)

    doc = Document(doc_id=f"{opts.doc_id_prefix}{name}", source=name, metadata=doc_meta)
    g = BlockGraph()
    g.add_document(doc)

    header_lines = []
    for k in ["From", "To", "Cc", "Date", "Subject", "Message-Id", "In-Reply-To"]:
        v = msg.get(k)
        if v is not None:
            header_lines.append(f"{k}: {v}")
    header_text = "\n".join(header_lines).strip()

    # Body extraction: prefer text/plain
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                body = part.get_content() or ""
                break
        if not body:
            # fallback to first text-ish part
            for part in msg.walk():
                if part.get_content_type().startswith("text/"):
                    body = part.get_content() or ""
                    break
    else:
        body = msg.get_content() or ""

    body = body.strip()[: opts.max_body_chars]

    header_block = Block(
        block_id=f"{doc.doc_id}#header",
        doc_id=doc.doc_id,
        type="email_header",
        text=header_text,
        anchors={"message_id": msg_id},
        metadata={"subject": subj},
    )
    body_block = Block(
        block_id=f"{doc.doc_id}#body",
        doc_id=doc.doc_id,
        type="paragraph",
        text=body,
        parent_id=header_block.block_id,
        anchors={"message_id": msg_id},
        metadata={"subject": subj},
    )

    g.add_block(header_block)
    g.add_block(body_block)
    g.add_edge(Edge(src=header_block.block_id, dst=body_block.block_id, type="contains"))

    # threading edges (dst is a placeholder unless the caller resolves it across corpus)
    thread_ref = in_reply_to or (refs.split()[-1] if refs else "")
    if thread_ref:
        g.add_edge(
            Edge(
                src=header_block.block_id,
                dst=f"email:msgid:{thread_ref}",
                type="reply_to",
                metadata={"in_reply_to": in_reply_to, "references": refs},
            )
        )

    g.build_indexes()
    return g

