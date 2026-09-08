"""Regression: a manifest that names its own watermark soft binding must not
also report a SynthID watermark from the generic vendor-token inference.

Microsoft Designer manifests sign as Microsoft, carry the InvisMark
``c2pa.watermarked`` action, and name their generation agent
"Azure OpenAI ImageGen". The OpenAI issuer token inside that agent name plus
the watermarked action used to satisfy the OpenAI SynthID-evidence rule,
double-counting one forensic mark as two pixel watermarks.
"""

from __future__ import annotations

from remove_ai_watermarks._internal.c2pa import c2pa_info_from_manifest_store

DESIGNER_STORE = {
    "active_manifest": "designer",
    "manifests": {
        "designer": {
            "signature_info": {"issuer": "Microsoft Corporation", "common_name": "Microsoft Corporation"},
            "claim_generator_info": [{"name": "Microsoft Responsible AI Provenance", "version": "1.0"}],
            "assertions": [
                {
                    "label": "c2pa.actions",
                    "data": {
                        "actions": [
                            {
                                "action": "c2pa.created",
                                "softwareAgent": {"name": "Azure OpenAI ImageGen"},
                                "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
                            },
                            {"action": "c2pa.watermarked"},
                        ]
                    },
                },
                {
                    "label": "c2pa.soft-binding",
                    "data": {
                        "alg": "com.microsoft.invismark.1",
                        "blocks": [{"value": "bf7a2993-cc1f-47e1-b1f0-cd8839aabb22"}],
                    },
                },
            ],
        }
    },
}


def test_named_watermark_soft_binding_suppresses_generic_synthid_evidence() -> None:
    info = c2pa_info_from_manifest_store(DESIGNER_STORE)
    assert info["ai_source_kind"] == "generated"
    assert info["soft_binding_algorithm"] == "com.microsoft.invismark.1"
    assert info.get("synthid_watermark") is None
    assert info.get("synthid_vendors") is None


def test_content_fingerprint_does_not_suppress_google_synthid_evidence() -> None:
    store = {
        "active_manifest": "google",
        "manifests": {
            "google": {
                "signature_info": {"issuer": "Google LLC"},
                "assertions": [
                    {
                        "label": "c2pa.actions",
                        "data": {
                            "actions": [
                                {
                                    "action": "c2pa.created",
                                    "digitalSourceType": (
                                        "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
                                    ),
                                }
                            ]
                        },
                    },
                    {"label": "c2pa.soft-binding", "data": {"alg": "io.iscc.v0"}},
                ],
            }
        },
    }

    info = c2pa_info_from_manifest_store(store)

    assert info["soft_binding"] == "ISCC (content code)"
    assert info["synthid_vendors"] == ["Google LLC"]
    assert info["synthid_watermark"] == "present according to Google LLC provenance"


def test_vendor_agent_name_alone_is_not_the_vendors_provenance() -> None:
    """The identity-scoped inference must not fire on a service name either.

    Same manifest without the soft binding: the "Azure OpenAI ImageGen" agent
    is not an OpenAI signature or claim generator, so no OpenAI SynthID
    evidence may be derived from it.
    """
    store = {
        "active_manifest": "designer",
        "manifests": {
            "designer": {
                "signature_info": {"issuer": "Microsoft Corporation", "common_name": "Microsoft Corporation"},
                "claim_generator_info": [{"name": "Microsoft Responsible AI Provenance", "version": "1.0"}],
                "assertions": [
                    {
                        "label": "c2pa.actions",
                        "data": {
                            "actions": [
                                {
                                    "action": "c2pa.created",
                                    "softwareAgent": {"name": "Azure OpenAI ImageGen"},
                                    "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
                                },
                                {"action": "c2pa.watermarked"},
                            ]
                        },
                    }
                ],
            }
        },
    }
    info = c2pa_info_from_manifest_store(store)
    assert info.get("synthid_watermark") is None
