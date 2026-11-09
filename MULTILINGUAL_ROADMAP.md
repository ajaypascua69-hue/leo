# Leo multilingual development plan

## Goal

Design Leo to become a multilingual general-purpose model, with a long-term target of 50+ languages. This is an engineering target, not a claim of current proficiency. Language support must be evaluated per language and task.

Initial priority coverage should include:
- English; Filipino/Tagalog.
- Japanese; Korean; Chinese (with script/locale distinctions where useful).
- Spanish; Portuguese; French; German.
- Arabic; Hindi; Indonesian; and additional languages selected using data availability, user needs, and responsible licensing.

Expand beyond this list toward 50+ languages, including lower-resource languages. Do not equate a language's presence in training data with reliable fluency.

## Tokenization and scripts

The current UTF-8 byte tokenizer can encode any Unicode text and does not depend on a pretrained vocabulary. It is a useful clean-room baseline, but byte-level sequences are long and can be inefficient for multilingual pretraining.

A future subword tokenizer must be trained from scratch on the approved multilingual corpus. It should:
- Preserve Unicode correctly and cover Latin, Cyrillic, Greek, Arabic, Indic, Hangul, Han, kana, and other scripts represented in the corpus.
- Avoid starving low-resource scripts through vocabulary allocation.
- Include normalization tests, combining marks, diacritics, punctuation, emoji, mixed-script text, and round-trip encode/decode tests.
- Record tokenizer version and corpus provenance with every checkpoint.
- Never import tokenizer files or vocabulary from a pretrained model.

## Data mixture and evaluation

- Record ISO 639 and BCP 47-style language tags at the document level.
- Audit document counts and token/byte counts by language before training.
- Build train/validation/test splits at document or source level to reduce leakage.
- Use a documented sampling strategy so high-resource languages do not silently dominate. Keep language weights configurable and publish them.
- Use only reviewed, properly licensed, traceable sources. Do not scrape private or paywalled content.
- Evaluate each language separately for next-token loss, reading comprehension, instruction following, translation, writing quality, and reasoning where appropriate.
- Include human review by competent speakers for quality claims; automated metrics alone are insufficient.
- Report low-data languages honestly and avoid claiming fluency based on a handful of examples.

## Milestones

1. **Metadata foundation:** accept language tags without a hard-coded two-language ceiling; report language distribution.
2. **Corpus foundation:** collect and review small, balanced, legally usable samples across priority languages.
3. **Tokenizer research:** compare byte baseline against a tokenizer trained from scratch.
4. **Baseline training:** run reproducible small-scale experiments and publish per-language results.
5. **Scale-up:** grow parameters, context, data, and compute only when evaluation supports the changes.
6. **Language expansion:** add languages in batches with data-quality and speaker-review gates.
