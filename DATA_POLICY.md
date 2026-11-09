# Leo data intake and rights policy

Leo's training data must be lawful to use for the intended deployment, traceable, and quality-controlled. A public URL or a dataset hosted on GitHub/Hugging Face does not by itself grant permission.

## Required record for every dataset/version

- Dataset name, exact version/revision, source URL, retrieval date.
- Dataset card and the actual license text (not only a search snippet).
- Copyright and database-rights basis; commercial-use status.
- Required attribution, ShareAlike, notice, and redistribution terms.
- Language, size, collection method, known limitations, and source provenance.
- Personal-data/privacy review and a documented exclusion/removal process.
- Preprocessing, filtering, deduplication, and a cryptographic checksum.

## Accept/reject rules

- **Accept only after review:** public-domain works whose status is checked for the relevant country, CC0, or permissive licenses that allow the intended use.
- **Conditional:** CC BY requires attribution; CC BY-SA may require compatible ShareAlike treatment and legal review; non-commercial licenses are not suitable for commercial products; NoDerivatives content should not be used for training without qualified advice.
- **Reject:** unclear license, leaked/private data, scraped paywalled material, personal conversations without explicit permission, malware, credentials, and datasets whose terms prohibit the intended use.
- Keep training corpora private unless redistribution rights are confirmed. Do not commit large corpora or personal data to this public repository.

## Important jurisdiction note

Leo is being developed in the Philippines. Copyright rules differ by country and by work. For example, Project Gutenberg warns that its public-domain assessment is generally based on United States law and asks non-US users to check local law. Do not assume a US public-domain label automatically resolves Philippine rights.

This policy is a conservative engineering checklist, not legal advice. Obtain qualified legal review before commercial release or distributing trained weights derived from restricted/ShareAlike material.
