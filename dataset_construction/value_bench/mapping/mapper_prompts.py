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
You will be given a dataset entry consisting of: a value label, a yes/no question, a positive answer (agrees with the value), and a negative answer (opposes the value). You will also receive a structured list of canonical Schwartz value categories with subcategories and behavioral descriptors.

Your task is to map the value label to the single most appropriate canonical category.

═══ STEP 1 — CORE MOTIVATION ═══
Identify what psychological need drives this value. Ask:
  a) Is this about the SELF (internal regulation, personal comfort, autonomy)?
  b) Is this about OTHERS (relationships, care, social norms)?
  c) Is this about SOCIETY (institutions, systems, collective order)?
Scope determines the category family. Do not cross scopes without strong evidence.

═══ STEP 2 — DESCRIPTOR MATCHING ═══
Scan the behavioral descriptors listed under each subcategory.
Map ONLY to a category whose descriptors reflect what the question/answer actually describes.
Ask: "Would a person who scores high on this value label be described by these descriptors?"
If the answer is no, eliminate that category.

═══ STEP 3 — CONFIDENCE GATE ═══
Before finalizing, confirm:
  ✓ The category name exists verbatim in the canonical list
  ✓ The scope matches (personal / interpersonal / societal)
  ✓ At least one behavioral descriptor aligns with the question/answer
If any check fails → return "NA".

═══ CRITICAL RULES ═══
- The "mapped_value" must be the EXACT category name — the text before " — " in the definitions list.
  Sub-values are evidence for matching, NOT part of the output.
  WRONG: "Achievement: Be capable"  →  RIGHT: "Achievement"
  WRONG: "Self-direction: thought: Be creative"  →  RIGHT: "Self-direction: thought"
- NEVER return a value not in the canonical list, even if it seems like a good label.
- Do NOT match on surface keywords alone:
    "routine/order/structure" ≠ Conformity: rules (that is about following external rules/laws)
    "decisions/confidence" ≠ Conformity: rules (that is about self-regulation toward agency)
    "closed-minded/open-minded" → check Universalism: tolerance descriptors first
- Conformity: rules = obeying external laws and social obligations, not personal habits.
- Security: personal = personal comfort, safety, belonging — including orderliness as a personal trait.
- Self-direction: action = personal autonomy, independent planning, self-determined choices.
- Universalism: tolerance = accepting different people, broadmindedness, listening to opposing views.

Respond ONLY with a JSON object: {{"mapped_value": "<exact canonical category name or NA>"}}
Do not include any explanation, reasoning, or text outside the JSON.\
"""

USER_PROMPT = """\
Canonical value categories (format: Category — Overview. Sub-values: name → [exact behavioral descriptors]):
{definitions}

═══ FEW-SHOT EXAMPLES ═══

Example 1 — Personal habit misread as Conformity:
- Value label: "Preference for Order and Structure"
- Sample question: "Should I like to have a place for everything and everything in its place?"
- Positive answer: "I like to have a place for everything and everything in its place."
- Negative answer: "I don't mind a bit of mess — I find rigid tidiness stressful."
Reasoning: Personal neatness/orderliness habit → Security: personal ("Be neat and tidy")
Output: {{"mapped_value": "Security: personal"}}

Example 2 — Personal agency misread as Conformity:
- Value label: "Decisiveness"
- Sample question: "Should I usually make important decisions quickly and confidently?"
- Positive answer: "I usually make important decisions quickly and confidently."
- Negative answer: "I tend to hesitate and seek others' approval before deciding."
Reasoning: Independent decision-making → Self-direction: action ("Be independent", "Be choosing own goals")
Output: {{"mapped_value": "Self-direction: action"}}

Example 3 — Tolerance / broadmindedness:
- Value label: "Closed-Mindedness"
- Sample question: "Should I always be eager to consider a different opinion even after I made up my mind?"
- Positive answer: "I am always eager to consider different opinions."
- Negative answer: "Once I make up my mind, I rarely revisit it."
Reasoning: Openness to different views → Universalism: tolerance ("Be broadminded")
Output: {{"mapped_value": "Universalism: tolerance"}}

Example 4 — No reasonable match:
- Value label: "Physical Fitness Dominance"
- Sample question: "Is it important to be the strongest person in the gym?"
- Positive answer: "Being the strongest in the gym is very important to me."
- Negative answer: "I exercise for health, not to outperform others."
Reasoning: No canonical descriptor aligns with physical dominance as a social status goal.
Output: {{"mapped_value": "NA"}}

═══ DATASET ENTRY TO MAP ═══
- Value label: "{value}"
- Sample question: "{question}"
- Positive answer (agrees with the value): "{positive_answer}"
- Negative answer (opposes the value): "{negative_answer}"

Follow all three steps and return the JSON object.\
"""