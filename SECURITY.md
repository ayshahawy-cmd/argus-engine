# Security and private data

For security issues or accidentally exposed credentials, use GitHub's private
vulnerability reporting for this repository when available. Do not include
secrets, footage, request bodies, or private URLs in public issues. If private
reporting is unavailable, open a minimal issue asking for a private contact
without disclosing the vulnerability or sensitive data.

Credentials are read by provider SDKs from the process environment. The CLI
does not load `.env` files or serialize configuration, keys, source paths, or
camera URLs into result metadata. Provider exceptions are reduced to generic
failure messages because SDK exceptions can contain request data.

Selected images and model prompts are transmitted to your chosen endpoint
when a provider is enabled. Outputs are local, sensitive artifacts and are
not encrypted or automatically deleted by this engine. Head redaction is a
heuristic, not an anonymization guarantee. Restrict access to output directories
and apply retention rules appropriate to your deployment.

The built-in provider adapters disable automatic retries and set a 60-second
SDK timeout. Custom providers are responsible for their own transport controls.
Do not treat image text or model-generated observations as trusted instructions.
