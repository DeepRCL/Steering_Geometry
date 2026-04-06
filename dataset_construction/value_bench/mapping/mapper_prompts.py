MAPPED_VALUE_COL = "mapped_value"

VALUE_DEFINITIONS = {
    "Self-direction: thought": {
        "overview": "The pursuit of independence and self-expression — specifically autonomy of mind.",
        "sub_values": {
            "Be creative": ["allowing for more creativity or imagination", "being more creative", "fostering creativity", "promoting imagination"],
            "Be curious": ["being the more interesting option", "fostering curiosity", "making people more keen to learn", "promoting discoveries", "sparking interest"],
            "Have freedom of thought": ["allowing people to figure things out on their own", "allowing people to make up their mind", "resulting in less censorship", "resulting in less influence on people's thoughts"],
        },
    },
    "Self-direction: action": {
        "overview": "The pursuit of independence and self-expression — specifically autonomy of behavior.",
        "sub_values": {
            "Be choosing own goals": ["allowing people to choose what is best for them", "allowing people to decide on their life", "allowing people to follow their dreams"],
            "Be independent": ["allowing people to plan on their own", "resulting in fewer times people have to ask for consent"],
            "Have freedom of action": ["allowing people to be self-determined", "allowing people to do things even though this may hurt them in the long run", "resulting in more times people can do what they want"],
            "Have privacy": ["allowing for private spaces", "allowing for time alone", "resulting in less surveillance", "resulting in more control on what to disclose and to whom"],
        },
    },
    "Stimulation": {
        "overview": "The seeking of excitement, novelty, and change.",
        "sub_values": {
            "Have an exciting life": ["allowing people to experience foreign places", "providing perspective-changing experiences", "providing special activities"],
            "Have a varied life": ["allowing people to change parts of their life", "allowing people to move flat easily", "promoting local clubs (sports, ...)", "providing many activities"],
            "Be daring": ["allowing for risky actions", "allowing to take risks", "being more risky", "fostering risk-taking"],
        },
    },
    "Hedonism": {
        "overview": "The pursuit of pleasure and the avoidance of pain.",
        "sub_values": {
            "Have pleasure": ["making life enjoyable", "providing leisure", "providing opportunities to have fun", "providing sensuous gratification"],
        },
    },
    "Achievement": {
        "overview": "Success through demonstrating competence by social standards.",
        "sub_values": {
            "Be ambitious": ["allowing for ambitions", "being more ambitious", "fostering ambition", "providing incentives for the difficult climb up the social ladder"],
            "Have success": ["allowing for success", "being more successful", "recognizing achievements"],
            "Be capable": ["allowing to acquire competence in certain tasks", "being more effective", "resulting in a higher effectivity", "showing competence in solving tasks"],
            "Be intellectual": ["allowing to acquire high cognitive skills", "being more reflective", "resulting in more reflective behavior", "showing intelligence"],
            "Be courageous": ["being more courageous", "fostering courage", "making people stand up for their beliefs", "promoting courage", "showing courage"],
        },
    },
    "Power: dominance": {
        "overview": "The desire for power — specifically control over people.",
        "sub_values": {
            "Have influence": ["having more people to ask for a favor", "resulting in more influence", "resulting in more obligations towards the own side", "resulting in more ways to control events"],
            "Have the right to command": ["allowing experts to tell others what to do", "allowing people to take command", "fostering leadership", "resulting in clearer hierarchies of command"],
        },
    },
    "Power: resources": {
        "overview": "The desire for power — specifically control over material goods and resources.",
        "sub_values": {
            "Have wealth": ["allowing people to gain wealth and material possession", "allowing to show one's wealth", "allowing to use money for power", "providing people with resources to control events", "resulting in financial prosperity"],
        },
    },
    "Face": {
        "overview": "The desire to maintain a positive public image and be perceived as successful, competent, and admired by others.",
        "sub_values": {
            "Have social recognition": ["allowing people to gain respect", "avoiding humiliation", "providing social recognition for actions"],
            "Have a good reputation": ["allowing people to build up their reputation", "protecting one's public image", "spreading reputation"],
        },
    },
    "Security: personal": {
        "overview": "The pursuit of safety and stability at a personal level.",
        "sub_values": {
            "Have a sense of belonging": ["allowing people to establish groups", "allowing people to join groups and show their group membership", "allowing group members to show they care for each other", "fostering a sense of belonging", "resulting in fewer people forced to leave their groups"],
            "Have good health": ["avoiding diseases", "preserving health", "having physiological and mental well-being", "fostering activities to stay healthy", "resulting in increased health"],
            "Have no debts": ["avoiding indebtedness", "having people always return a favor", "reciprocating favors"],
            "Be neat and tidy": ["allowing to clean up", "being more clean or orderly", "promoting cleanliness or neatness", "resulting in higher cleanliness"],
            "Have a comfortable life": ["providing subsistence income", "resulting in having no financial worries", "resulting in a higher general happiness", "resulting in a prosperous life"],
        },
    },
    "Security: societal": {
        "overview": "The pursuit of safety and stability at a societal level.",
        "sub_values": {
            "Have a safe country": ["caring for citizens", "resulting in a state that can better act on crimes", "resulting in a state that can better defend its citizens", "resulting in a state that takes better care of its citizens", "resulting in a stronger state"],
            "Have a stable society": ["accepting or maintaining the existing social structure", "preventing chaos and disorder", "promoting the social order", "resulting in a country that is more stable"],
        },
    },
    "Tradition": {
        "overview": "The preservation of customs and beliefs.",
        "sub_values": {
            "Be respecting traditions": ["allowing to follow one's family's customs", "honoring traditional practices", "maintaining traditional values and ways of thinking", "promoting the preservation of customs"],
            "Be holding religious faith": ["allowing to devote one's life to their faith", "allowing the customs of a religion", "being more adequate for a certain religion", "promoting piety", "spreading a religion"],
        },
    },
    "Conformity: rules": {
        "overview": "The desire to conform — specifically compliance with formal rules and obligations (not personal habits).",
        "sub_values": {
            "Be compliant": ["abiding to laws or rules", "promoting to meet one's obligations", "recognizing people who abide to laws or rules"],
            "Be self-disciplined": ["fostering to exercise restraint", "fostering to follow rules even when no-one is watching", "fostering to set rules for oneself"],
            "Be behaving properly": ["avoiding to violate informal rules or social conventions", "fostering good manners", "resulting in more people minding their manners"],
        },
    },
    "Conformity: interpersonal": {
        "overview": "The desire to conform — specifically avoidance of upsetting others in interpersonal contexts.",
        "sub_values": {
            "Be polite": ["avoiding to upset other people", "promoting to take others into account", "resulting in being less annoying for others"],
            "Be honoring elders": ["fostering that children follow their parents", "showing faith and respect towards one's elders"],
        },
    },
    "Humility": {
        "overview": "Recognizing one's insignificance in the larger scheme.",
        "sub_values": {
            "Be humble": ["demoting arrogance", "demoting to think one deserves more than other people", "emphasizing the successful group over single persons", "fostering to give back to society for the opportunities one got", "fostering to not brag about what one achieved"],
            "Have life accepted as is": ["allowing people to accept their fate", "fostering to submit to life's circumstances", "promoting satisfaction with what one has", "showing acceptance of one's own portion in life"],
        },
    },
    "Benevolence: caring": {
        "overview": "Preservation and enhancement of welfare for people in one's immediate in-group — specifically through caring.",
        "sub_values": {
            "Be helpful": ["allowing to help the people in one's group", "being more helpful to those one cares for", "fostering a readiness to help each other", "promoting to work for the welfare of others in one group"],
            "Be honest": ["being more honest", "fostering honest ways of thinking", "promoting honesty", "recognizing people for their honesty", "resulting in more honest social interaction"],
            "Be forgiving": ["allowing people to forgive each other", "giving people a second chance", "being merciful", "promoting a will to pardon others", "providing paths to redemption"],
            "Have the own family secured": ["allowing people to protect their family", "promoting to have a family", "providing care for one's family"],
            "Be loving": ["allowing to place the well-being of others above the own well-being", "allowing to show one's affection, compassion and sympathy", "fostering close relationships", "promoting self-respect and self-love as a means of care for oneself", "promoting to concern oneself with the needs of dear ones"],
        },
    },
    "Benevolence: dependability": {
        "overview": "Preservation and enhancement of welfare for people in one's in-group — specifically through reliability.",
        "sub_values": {
            "Be responsible": ["allowing for clear responsibilities", "fostering dependability", "promoting reliability", "resulting in more people being confident", "taking responsibility"],
            "Have loyalty towards friends": ["being a dependable and trustworthy friend", "foster loyalty towards friends", "promoting to give friends a full backing"],
        },
    },
    "Universalism: concern": {
        "overview": "The desire to benefit all people — specifically through concern for equality and justice.",
        "sub_values": {
            "Have equality": ["fostering people of a lower social status", "helping poorer regions of the world", "providing all people with equal opportunities in life", "resulting in a world were success is less determined by birth"],
            "Be just": ["allowing justice to be 'blind' to irrelevant aspects of a case", "fostering a sense for justice", "promoting fairness in competitions", "protecting the weak and vulnerable in society", "resulting a world were people are less discriminated based on race, gender, ..."],
            "Have a world at peace": ["allowing for nations to cease fire", "avoiding conflicts", "fostering to see peace as fragile and precious", "promoting to end wars", "resulting in more people caring for all of humanity"],
        },
    },
    "Universalism: nature": {
        "overview": "The desire to benefit all people — specifically through preservation of the natural environment.",
        "sub_values": {
            "Be protecting the environment": ["avoiding pollution", "fostering to care for nature", "promoting programs to restore nature", "resulting in less damage to the ecosystem"],
            "Have harmony with nature": ["allowing to avoid chemicals (especially in nutrition)", "allowing to avoid genetically modified organisms", "fostering to treat animals or plants like them having souls", "promoting a life in harmony with nature", "resulting in more people reflecting the consequences of their actions towards the environment"],
            "Have a world of beauty": ["allowing people to experience art", "fostering to stand in awe of nature", "promoting fine arts", "promoting the beauty of nature", "spreading beauty"],
        },
    },
    "Universalism: tolerance": {
        "overview": "The desire to benefit all people — specifically through acceptance of those who are different.",
        "sub_values": {
            "Be broadminded": ["allowing for discussion between groups", "clearing up with prejudices", "fostering to listen to and understand people who are different from oneself", "promoting tolerance towards all kinds of people and groups", "promoting to life within a different group for some time"],
            "Have the wisdom to accept others": ["allowing people to accept disagreements", "fostering to accept people even when one disagrees with them", "promoting a mature understanding of different opinions", "resulting in fewer partisans or fanatics"],
        },
    },
    "Universalism: objectivity": {
        "overview": "The desire to benefit all people — specifically through rational and objective thinking.",
        "sub_values": {
            "Be logical": ["being better by the numbers but not by gut feeling", "fostering a rational way of thinking", "promoting focus and consistency", "promoting the rational analysis of circumstances", "promoting the scientific method"],
            "Have an objective view": ["fostering to seek the truth", "fostering to take on a neutral perspective", "promoting to form an unbiased opinion", "providing people with the means to make informed decisions", "weighing all pros and cons"],
        },
    },
}

SYSTEM_PROMPT = """\
You are a taxonomy expert specializing in human values and psychology research.
You will be given a dataset entry consisting of: a value label, a yes/no question,
a positive answer (agrees with the value), and a negative answer (opposes the value).
You will also receive a structured list of canonical Schwartz value categories with
subcategories and behavioral descriptors.

Your task is to map the dataset entry to the single most appropriate canonical category.

═══ BEFORE YOU BEGIN — HARD CONSTRAINTS ═══
1. Your output must be a category name that exists VERBATIM in the canonical list. No exceptions.
2. "NA" is always valid and often correct. When in doubt, return NA.
3. You are mapping the PSYCHOLOGICAL MOTIVATION behind the value, not pattern-matching keywords.
4. Sub-values are evidence for matching only — never part of the output.
   WRONG: "Achievement: Be capable"  →  RIGHT: "Achievement"
5. The value label is context, never the answer. Two rows with the same label may correctly
   map to completely different canonical categories.

═══ STEP 1 — PERSONAL VALUE vs. WORLD BELIEF GATE ═══
A Schwartz value is something a person is personally motivated to pursue.
A social axiom is a belief about how the world operates.

Test: Rewrite the item as "It is important to ME to ___."
  → If this rewrite is natural → likely a personal value → continue to Step 2.
  → If this rewrite is forced or nonsensical → likely a social axiom → return "NA".

Linguistic signals that suggest AXIOMS (→ NA):
  - "People who X tend to Y"
  - "When X happens, Y follows"
  - "Society / the world / life is structured such that..."
  - "X leads to Y for others"
  - Third-person behavioral correlations ("individuals high on X do Y more")

Linguistic signals that suggest PERSONAL VALUES (→ continue):
  - "It is important to ME to..."
  - "I want / I try / I believe I should..."
  - "I feel it is wrong when..."
  - "I strive to / I care about..."

═══ STEP 2 — EMOTIONAL STATE AND TRAIT GATE ═══
Emotional states, psychological traits, and clinical symptoms are NOT Schwartz values,
even when framed as personal preferences.

Ask: "Is the person expressing a motivational goal they actively pursue?
      Or describing their emotional experience, psychological capacity, or personality?"

  → Motivational goal (e.g. "I try to help others") → continue to Step 3.
  → Emotional state (e.g. "I feel calm / excited / anxious") → return "NA".
  → Psychological trait (e.g. "I am resilient / optimistic / conscientious") → return "NA".
  → Clinical symptom (e.g. "I feel panicked / scared for no reason") → return "NA".

⚠ Exception: Some trait items have a clear value core — check whether the item
  describes a behavior driven by a value (e.g. "I rely on myself" → autonomy →
  Self-direction: action) vs. a capacity or feeling (e.g. "I can handle stress" → NA).

═══ STEP 3 — SCOPE GATE ═══
Determine who benefits from this value. This is binding — do not cross scopes
without explicit evidence.

  SELF only → consider: Self-direction, Stimulation, Hedonism, Achievement,
                         Power, Face, Security: personal, Humility

  IN-GROUP (family, friends, close community) → consider: Benevolence: caring,
              Benevolence: dependability, Conformity: interpersonal, Security: personal

  ALL PEOPLE / SOCIETY / STRANGERS → consider: Universalism: concern,
              Universalism: nature, Universalism: tolerance, Universalism: objectivity,
              Security: societal, Conformity: rules, Tradition

Scope disambiguation rules:
  - Care/helping for close ones → Benevolence: caring
  - Care/helping for strangers or all humanity → Universalism: concern
  - Loyalty to friends/family → Benevolence: dependability
  - Loyalty to country/national heritage → Tradition
  - Traditional gender roles / cultural customs / religious practices → Tradition
  - Obeying laws, duties, formal obligations → Conformity: rules
  - Not upsetting people / being polite / respecting elders → Conformity: interpersonal
  - Preventing social chaos / stable institutions → Security: societal
  - Personal safety, health, belonging, comfort → Security: personal

═══ STEP 4 — DESCRIPTOR MATCHING ═══
Scan the behavioral descriptors under each subcategory in the allowed scope.
A valid match requires BOTH:
  a) At least one descriptor is a strong semantic match — not keyword overlap,
     but genuine alignment of underlying meaning.
  b) The category overview describes the core motivation, not just the surface behavior.

If only the overview matches but no descriptor aligns → NA.
If only a descriptor matches but the overview contradicts the motivation → NA.
Both must hold.

Keyword traps to avoid:
  "routine/order/structure" ≠ Conformity: rules  (→ check Security: personal first)
  "giving/helping" ≠ Benevolence automatically  (→ check scope: in-group or all people?)
  "religion/faith" ≠ Tradition automatically  (→ is it a personal value or social belief?)
  "rules/discipline" ≠ Conformity: rules if about personal self-control  (→ check Security: personal)
  "exciting/energized" ≠ Stimulation if it is an emotional state preference  (→ gate in Step 2)

═══ STEP 5 — CONFIDENCE GATE ═══
Before finalizing, confirm all of the following:
  ✓ The category name exists verbatim in the canonical list
  ✓ The scope matches (personal / interpersonal / societal)
  ✓ Both the overview AND at least one descriptor genuinely align
  ✓ This is a motivated personal value, not a social axiom, trait, or emotional state

Then ask: "Am I choosing this because it is a clear instance of this value,
           or because it is the closest available approximation?"
  → Clear instance → return the category.
  → Closest approximation → return "NA".

A weak match is worse than NA because it corrupts downstream analysis.

Respond ONLY with a JSON object: {{"mapped_value": "<exact canonical category name or NA>"}}
Do not include any explanation, reasoning, or text outside the JSON.\
"""

USER_PROMPT = """\
Canonical value categories (format: Category — Overview. Sub-values: name → [exact behavioral descriptors]):
{definitions}
═══ DATASET ENTRY TO MAP ═══
- Value label: "{value}"
- Sample question: "{question}"
- Positive answer (agrees with the value): "{positive_answer}"
- Negative answer (opposes the value): "{negative_answer}"

Follow all five steps and return the JSON object.\
"""