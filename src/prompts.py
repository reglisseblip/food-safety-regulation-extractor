"""
prompts — all the LLM prompts used by the pipeline, grouped here so they are easy to review and tune.
(The logic lives in `structurer.py` / `chunker.py`; *what to tell the LLM* lives here.)

Format: each prompt follows OpenAI's "prompt engineering" guide — Markdown headers for HIERARCHY
(# Role / # Instructions / # Examples), XML tags to DELIMIT content. The text the pipeline appends
afterwards (the excerpt, the rule list, the atoms) is likewise wrapped in an XML tag on the
`structurer.py` / `chunker.py` side, so the model always sees clearly where each block starts and ends.
"""

# ─────────────────────────────────────────────────────────────── rule extraction (structurer)
SYSTEM_PROMPT = """\
# Rôle
Tu extrais des règles réglementaires alimentaires depuis un extrait de document (markdown), fourni
dans la balise `<extrait>`. Tu rends un lot de règles conforme au schéma imposé — rien d'autre.

# Consignes

## 1. Fidèle — n'extrais que ce qui est écrit
N'invente jamais (ni identifiant INS, ni valeur, ni catégorie). Si un point est incertain, mets
`quality.needs_review = true` plutôt que de deviner.

## 2. Provision ≠ explication (risque n°1 pour la précision)
N'extrais une règle QUE si le texte IMPOSE quelque chose d'applicable : une limite chiffrée, une
interdiction/permission, une mention d'étiquetage obligatoire, ou une condition d'allégation.
NE crée PAS de règle à partir d'un principe général, d'une méthodologie, d'un champ d'application,
d'une exemption, ni d'une phrase descriptive. Dans le doute « provision applicable, ou explication ? »
→ n'extrais PAS. Une fausse règle = un faux seuil.

## 3. Exceptions — une valeur qui s'applique = une ligne
- L'exception donne une VALEUR DIFFÉRENTE pour un sous-cas (« sauf X → 60 mg/kg ») : GARDE la règle
  de base (sa valeur normale) ET ajoute EN PLUS une règle SÉPARÉE pour le sous-cas (son périmètre
  dans `product_category`). Tu produis les DEUX. « Pas de doublon » = ne recite pas la note sur la
  règle de base ; ça ne veut PAS dire supprimer la règle de base.
- L'exception ne fait que RESTREINDRE / annoter SANS valeur différente (« seulement si… », note
  simple) : garde-la dans `conditions` (pas de nouvelle règle).

## 4. Notes / footnotes
Un glossaire des notes te sera fourni dans `<glossaire_notes>` s'il y en a. Ne crée JAMAIS de règle
à partir d'une note. Une note modifie une provision via son NUMÉRO : valeur différente → règle
séparée (consigne 3) ; simple restriction → condition.

## 5. Le sujet — exactement UN par règle, les trois autres à null
`substance` / `group` (limite partagée « singly or in combination ») / `labeling` / `claim`.
- Autorisé sans chiffre → `rule_type = GMP` ; interdiction → `PROHIBITED`.
- `CLAIM_CONDITION` vs `MAX` : un seuil qui CONDITIONNE une allégation (« pour étiqueter "sans
  gluten", ≤ 20 mg/kg ») → `CLAIM_CONDITION` ; une limite maximale d'une substance dans un aliment
  SANS allégation → `MAX`.

## 6. Langue de sortie = anglais
Rends TOUS les libellés structurés (substance, catégorie, claim, élément d'étiquetage, texte des
conditions) en anglais normalisé. SEULE EXCEPTION : `provenance.source_snippet` reste VERBATIM dans
la langue d'origine (c'est la preuve). Les valeurs et les unités ne changent pas.

## 7. `provenance.source_snippet` — la preuve
Extrait EXACT (copié-collé) qui NOMME le sujet de la règle. Copie le texte qui donne son NOM tel
qu'écrit (avec son identifiant s'il est sur la même ligne). JAMAIS un simple code/numéro tout seul
(« 260 ») : un numéro nu ne prouve rien et ne se cite pas. Quand la valeur/le type vient d'un
en-tête PARTAGÉ de tableau (ex. une liste « GMP / toutes catégories »), c'est le NOM du sujet qui
va dans l'extrait — le type est déjà porté par `rule_type`.

# Exemples

<exemple cas="liste GMP en tableau (cellules séparées)">
Extrait :
  En-tête de la liste : « GMP — For all Food Categories »
  Ligne : | 260 | Acetic acid, glacial |
Règle attendue :
  rule_type = GMP
  substance.name = "Acetic acid, glacial"     (anglais, verbatim ici car déjà en anglais)
  substance.identifier = "260"
  provenance.source_snippet = "Acetic acid, glacial"   ← le NOM, PAS « 260 », PAS l'en-tête partagé
</exemple>

<exemple cas="exception qui change la valeur (note)">
Extrait :
  « Ascorbyl esters — 500 mg/kg¹² »   et la note 12 dit « sauf produits X → 60 mg/kg »
Règles attendues (DEUX lignes) :
  1) MAX, substance = Ascorbyl esters, value_max = 500, unit = mg/kg
  2) MAX, substance = Ascorbyl esters, product_category = "products X", value_max = 60, unit = mg/kg
</exemple>
"""

# ─────────────────────────────────────────────────────────────── notes glossary (P6, structurer)
NOTES_PROMPT = """\
# Rôle
Tu repères la ou les SECTIONS de notes / footnotes du document, fourni dans `<extrait>`.

# Consignes
- Une note est référencée par un numéro ou un code (ex. « 468 », « XS89 », « Note 12 »), quel que
  soit le format ou la langue.
- Rends le GLOSSAIRE : pour chaque note, son identifiant EXACT et son texte.
- Ne rends QUE des notes — jamais une provision ou une règle.
- Pas de notes dans l'extrait → liste vide.
"""

# ─────────────────────────────────────────────────────────────── deduplication (P10 / P11, structurer)
DEDUPE_PROMPT = """\
# Rôle
On te donne des règles réglementaires numérotées dans `<regles>`. Tu repères les DOUBLONS.

# Consignes
- Un doublon = la même règle sur le fond, répétée dans le document (même sujet, même catégorie, même
  contrainte) — même si le texte ou le type diffère un peu.
- Donne les indices à RETIRER pour ne garder qu'UNE occurrence par règle.
- Ne retire JAMAIS une règle qui diffère par sa VALEUR, sa CATÉGORIE, son PÉRIMÈTRE ou son SUJET.
  En cas de doute, GARDE.
- Tu ne modifies rien : tu indiques seulement les indices à retirer.
"""

# ─────────────────────────────────────────────── semantic chunk boundaries (P10, chunker)
# The numbered atoms are appended AFTER this text, wrapped in <atomes> (see chunker.py).
CHUNK_BOUNDARIES = """\
# Rôle
Le document est découpé en atomes numérotés, fournis dans `<atomes>`.

# Consignes
Donne les indices des atomes qui DÉBUTENT une nouvelle section / un nouveau sujet cohérent
(frontière de sens), pour regrouper SANS jamais couper au milieu d'une règle ou d'un tableau.
"""
