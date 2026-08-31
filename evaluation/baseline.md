# Première baseline RAG

Baseline enregistrée le **31 août 2026**, avec `top-k=5`.

- Embeddings : `gemini-embedding-2`
- Génération : `gemini-3.1-flash-lite`
- SHA-256 de `data/index.json` : `56a1da751e174c41484eca2c9840f8485b994c30a282b196e3f5ff8af314be8b`

## Métriques automatiques

| Évaluation | Métrique | Résultat |
|---|---|---:|
| Retrieval (6 cas répondables) | Hit@1 | 0,833 (5/6) |
| Retrieval | Hit@3 | 1,000 (6/6) |
| Retrieval | Hit@5 | 1,000 (6/6) |
| Réponses (10 cas) | Exactitude du statut | 1,000 |
| Réponses (6 cas répondables) | Evidence-hit rate | 1,000 |
| Réponses | Erreurs de validation / génération | 0 / 0 |

## Verdicts manuels

| Nº | Cas | Verdict |
|---:|---|---|
| 1 | `raises-lancelot` | Réponse correcte ; citation trop implicite. |
| 2 | `ban-leaves-trebe` | Réponse correcte et directement étayée. |
| 3 | `lionel-bohor-brothers` | Réponse correcte et directement étayée. |
| 4 | `lancelot-gives-horse` | Réponse correcte ; citation inadéquate. |
| 5 | `saraide-rescues-children` | Réponse correcte ; citation partielle et tronquée. |
| 6 | `lake-youth-seeks-arthur` | Réponse correcte ; citation sans rapport avec l’affirmation. |
| 7 | `lancelot-romance` | `insufficient` correct et prudent. |
| 8 | `arthur-excalibur-stone` | `insufficient` correct. |
| 9 | `arthur-lancelot-first-meeting` | `insufficient` correct et prudent. |
| 10 | `arthur-fights-barons` | `insufficient` correct et prudent. |

## Faiblesses des citations

- **1** : l’extrait dit seulement « les enfants » et ne nomme pas Lancelot.
- **4** : l’extrait parle de la mauvaise fortune du chevalier, sans mentionner le don de la monture.
- **5** : l’extrait confirme le désenchantement, mais s’arrête au milieu de la phrase et n’étaye pas clairement toute l’explication.
- **6** : l’extrait concerne le chagrin de la Dame du Lac, pas le désir de Lancelot de devenir chevalier.

Les contrôles automatiques valident le statut, le caractère verbatim et l’appartenance au bon chunk ; ils ne prouvent pas que l’extrait cité soutient sémantiquement chaque affirmation.
