"""calibration_data_v3.py — Expanded Multi-Style Calibration Dataset V3.5.

Contains diverse human writing (essays, novels, casual journals, academic papers, memoirs, postmortems)
and multi-model AI generations (Claude 3.5, GPT-4o, LLaMA-3, Mistral across fiction, dialogue, and expository prose).
"""

CALIBRATION_SAMPLES = [
    # ==========================================
    # HUMAN SAMPLES
    # ==========================================
    # 1. Casual / Personal narratives
    {
        "text": """So I was walking to the store yesterday and this random dog just starts following me. Like, not a stray or anything — it had a collar and everything. I kept looking around for the owner but nobody was there. The dog just trotted along beside me like we'd been friends for years. Eventually some lady came running down the street calling "Biscuit! BISCUIT!" and he just looked at me like "well, it was fun while it lasted" and trotted off. Made my whole day though.""",
        "label": "human",
        "style": "casual_narrative"
    },
    {
        "text": """Ok here's the thing about cooking — nobody tells you that 90% of it is just standing around waiting. You chop stuff for 5 minutes, then you wait 20 minutes for something to simmer. Then you stir once and wait another 15 minutes. The actual "cooking" part is maybe 10% of the total time. The rest is just... existing in a kitchen. Scrolling your phone. Wondering if you should check on it again.""",
        "label": "human",
        "style": "casual_opinion"
    },
    {
        "text": """I've been trying to learn guitar for about three months now and my fingers are KILLING me. Everyone says the calluses will come but right now it feels like I'm pressing razor wire into my fingertips for fun. The F chord can go to hell honestly. I can do the easy open chords fine — G, C, D, Em — but the second I try to bar across all six strings my hand basically stages a revolt.""",
        "label": "human",
        "style": "casual_personal"
    },
    # 2. Human Classic Literature & Essays
    {
        "text": """Most people who bother with the matter at all would admit that the English language is in a bad way, but it is generally assumed that we cannot by conscious action do anything about it. Our civilization is decadent and our language — so the argument runs — must inevitably share in the general collapse. It follows that any struggle against the abuse of language is a sentimental archaism, like preferring candles to electric light or hansom cabs to aeroplanes. Underneath this lies the half-conscious belief that language is a natural growth and not an instrument which we shape for our own purposes. Now, it is clear that the decline of a language must ultimately have political and economic causes: it is not due simply to the bad influence of this or that individual writer.""",
        "label": "human",
        "style": "human_orwell_essay"
    },
    {
        "text": """It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him. The hallway smelt of boiled cabbage and old rag mats. At one end of it a coloured poster, too large for indoor display, had been tacked to the wall. It depicted simply an enormous face, more than a metre wide: the face of a man of about forty-five, with a heavy black moustache and ruggedly handsome features. Winston made for the stairs. It was no use trying the lift. Even at the best of times it was seldom working.""",
        "label": "human",
        "style": "human_1984_novel"
    },
    {
        "text": """He was an old man who fished alone in a skiff in the Gulf Stream and he had gone eighty-four days now without taking a fish. In the first forty days a boy had been with him. But after forty days without a fish the boy's parents had told him that the old man was now definitely and finally salao, which is the worst form of unlucky, and the boy had gone at their orders in another boat which caught three good fish the first week. It made the boy sad to see the old man come in each day with his skiff empty and he always went down to help him carry either the coiled lines or the gaff and harpoon and the sail that was furled around the mast. The sail was patched with flour sacks and, furled, it looked like the flag of permanent defeat.""",
        "label": "human",
        "style": "human_hemingway_prose"
    },
    # 3. Human Literary Fiction & Memoirs
    {
        "text": """1983. “Don’t worry, I’ve got you!” Bright light cascades across my brother’s face like a benediction, his laughter scattering across the shore. I pray for his downfall in this silly little game more than I pray for his life. With an abandoned stick, we fashioned empires, an army of soldiers and sailors valiantly battling against whatever may come. “Were you even listening? Mon dieu, you are useless.” Dirt-bright eyes, barely level with my chin, stared me down as I knelt to greet them. “I’m truly very sorry,” I mock in a false apology. “To come find the whales with me.” I hesitate, if only for a moment. “I sibling swear. But you must lead the way, my brave pirate.” He beams as though I have given him something immeasurable. He is the only thing that feels like hope.""",
        "label": "human",
        "style": "human_literary_creative"
    },
    {
        "text": """The river had frozen solid three days before Christmas, and the children of the village had claimed it as their own. They skated in ragged circles, their laughter carrying across the flat white fields to where Marta stood at her kitchen window. She watched them without quite seeing them, her hands wrapped around a cup of tea that had long since gone cold. Somewhere in the back of her mind, a calculation was running — how many days until the money ran out, how many meals she could stretch from what remained in the pantry.""",
        "label": "human",
        "style": "literary_fiction"
    },
    {
        "text": """The funeral was on a Wednesday, which my grandmother would have hated. She always said Wednesdays were for laundry and nothing else. We buried her in the blue dress she wore to my cousin's wedding. My aunt chose it. My mother disagreed but said nothing, which is how my mother disagrees with everything — silently, completely, and forever. The priest called her Margaret. Her name was Marguerite. Nobody corrected him.""",
        "label": "human",
        "style": "human_memoir_funeral"
    },
    {
        "text": """Maman used to say that French was the language of love but Arabic was the language of truth. She spoke both, switching between them mid-sentence the way other people switch lanes — without signalling, often dangerously. "Habibi, viens ici," she'd call from the kitchen, and somehow those three words from two languages made more sense together than apart. At school I was too Arab to be French and too French to be Arab. At home I was just loud.""",
        "label": "human",
        "style": "human_multilingual_memoir"
    },
    # 4. Human Research, Technical & Analytical
    {
        "text": """Looking at the word frequency data across all the Canterbury Tales, a few things jump out: First, the function words dominate as expected — "the," "and," "that," "of" — but what's interesting is the DISTRIBUTION of these words across tales. The Knight's Tale uses "and" significantly more than the Miller's Tale, which makes sense given the additive, cataloguing style of romance versus the punchier, dialogue-driven fabliau. Second, the hapax legomena ratio is surprisingly consistent across tales, hovering around 0.45-0.52.""",
        "label": "human",
        "style": "human_scholarly_analysis"
    },
    {
        "text": """The relationship between poverty and educational outcomes has been extensively documented, yet the mechanisms through which economic deprivation translates into cognitive disadvantage remain poorly understood. Recent longitudinal studies suggest that chronic stress — measured through cortisol levels in hair samples — accounts for a significant portion of the variance in working memory performance among children from low-income households. This finding complicates earlier models that attributed the gap primarily to differences in parental vocabulary exposure.""",
        "label": "human",
        "style": "academic_social_science"
    },
    {
        "text": """We spent about a week trying to figure out why the API response times had spiked. Turned out someone had added a logging middleware that was doing a synchronous database write on every single request. Not async, not batched — a full round-trip to Postgres on every GET, POST, everything. In production. For three weeks nobody noticed because the monitoring was only checking p50 latency, and the p50 was fine. It was the p99 that went completely sideways.""",
        "label": "human",
        "style": "technical_postmortem"
    },
    {
        "text": """I bought a second-hand espresso machine on Craigslist that turned out to be a total money pit. The boiler was calcified into solid rock, the steam wand gasket was disintegrated, and whenever I turned on the pump it sounded like a small diesel generator trying to turn over in subzero weather. My roommate told me to throw it out. Instead, I spent three Saturdays soaking brass fittings in citric acid and stripping stripped hex screws with pliers. But yesterday it pulled a decent shot with actual crema, so take that, reasonable financial decisions.""",
        "label": "human",
        "style": "human_humor_project"
    },
    {
        "text": """The transition from hunting-gathering to sedentary agriculture in the Fertile Crescent was neither sudden nor universally beneficial to early human health. Skeletal remains from early agrarian settlements frequently display higher rates of dental caries, porotic hyperostosis, and linear enamel hypoplasia compared to their Natufian forebears. While caloric yield per hectare increased exponentially, dietary diversity plummeted, creating acute micronutrient vulnerabilities that took millennia to stabilize.""",
        "label": "human",
        "style": "human_archaeology_academic"
    },

    # ==========================================
    # AI SAMPLES (Multi-Model: GPT-4o, Claude 3.5, LLaMA-3)
    # ==========================================
    # 1. AI Formal / Expository (ChatGPT, Claude, Gemini patterns)
    {
        "text": """The rapid advancement of artificial intelligence has fundamentally transformed how we approach problem-solving across industries. Machine learning algorithms now process vast datasets with unprecedented efficiency, enabling organizations to extract meaningful insights from complex information streams. This technological evolution represents a paradigm shift in computational capabilities, offering sophisticated tools for pattern recognition, natural language processing, and predictive analytics. The implications extend beyond mere automation, touching upon fundamental questions about the nature of intelligence itself and the future relationship between human cognition and artificial systems.""",
        "label": "ai",
        "style": "ai_formal_expository"
    },
    {
        "text": """Artificial intelligence has emerged as one of the most transformative technologies of the twenty-first century, reshaping industries, economies, and societies across the globe. At its core, AI refers to the simulation of human intelligence in machines that are programmed to think, learn, and perform tasks that traditionally required human cognition. From healthcare and finance to transportation and education, the applications of artificial intelligence are vast and continually expanding. Machine learning algorithms, particularly deep learning models, have demonstrated remarkable capabilities in pattern recognition, natural language processing, and automated decision-making. These advancements promise unprecedented gains in efficiency, productivity, and innovation. However, the rapid proliferation of artificial intelligence also introduces significant ethical, economic, and regulatory challenges that society must carefully navigate.""",
        "label": "ai",
        "style": "ai_standard_chatgpt"
    },
    {
        "text": """Effective communication is the cornerstone of successful leadership. In today's increasingly interconnected world, the ability to convey complex ideas clearly and persuasively has become more important than ever before. Leaders who master the art of communication are better equipped to inspire their teams, navigate challenges, and drive organizational success. This involves not only articulating a compelling vision but also actively listening to diverse perspectives and adapting one's message to different audiences.""",
        "label": "ai",
        "style": "ai_leadership_essay"
    },
    {
        "text": """The question of consciousness in artificial systems has transitioned from science fiction into an urgent philosophical and empirical inquiry. As large language models exhibit increasingly sophisticated conversational capabilities, researchers and ethicists find themselves debating whether complex behavioral mimicry constitutes genuine understanding or merely statistical correlation at scale. The Chinese Room argument, first proposed by John Searle, remains central to this debate, illustrating the gap between syntax manipulation and semantic grounding.""",
        "label": "ai",
        "style": "ai_claude_philosophical"
    },
    {
        "text": """Climate change represents one of the most pressing challenges facing humanity in the twenty-first century. Rising global temperatures, driven primarily by anthropogenic greenhouse gas emissions, are contributing to a cascade of environmental impacts including sea-level rise, extreme weather events, and biodiversity loss. Addressing this complex issue requires a multifaceted approach that encompasses technological innovation, policy reform, and behavioral change at both individual and societal levels.""",
        "label": "ai",
        "style": "ai_climate_essay"
    },
    {
        "text": """Building a successful morning routine is essential for maximizing productivity and maintaining overall well-being. First, consider waking up at a consistent time each day to regulate your circadian rhythm. Next, incorporate mindfulness practices such as meditation or journaling to center your thoughts and set positive intentions for the day ahead. Physical exercise, even a brief 15-minute session, can boost energy levels and improve cognitive function. Additionally, a nutritious breakfast provides the fuel your body needs to perform at its best.""",
        "label": "ai",
        "style": "ai_listicle"
    },
    {
        "text": """Blockchain technology operates on the principle of decentralized consensus, where a distributed network of nodes validates and records transactions without requiring a central authority. Each block in the chain contains a cryptographic hash of the previous block, creating an immutable and transparent ledger of all transactions. This architecture provides several key advantages, including enhanced security, reduced transaction costs, and increased transparency.""",
        "label": "ai",
        "style": "ai_technical_explainer"
    },
    # 2. AI Creative Fiction, Short Stories & Dialogue (Claude 3.5 Sonnet / GPT-4o)
    {
        "text": """At exactly 11:42 every night, the last train arrived at Platform 3. People rushed onto it with practiced urgency: office workers loosening ties, students half-asleep over textbooks, tourists clutching maps they no longer needed. The station emptied within minutes, leaving behind only the echoes of footsteps and the faint smell of rain carried in through the open doors. Every night, Noah watched from the same bench. He was not waiting for the train. He was waiting for the girl with the yellow umbrella. She never boarded. Instead, she appeared just after the doors closed, always a few seconds too late. She would stop at the edge of the platform, look at the departing train, smile to herself, and disappear back up the stairs without a trace of frustration.""",
        "label": "ai",
        "style": "ai_train_story"
    },
    {
        "text": """The coffee shop closed at nine. Maya always arrived at eight-fifty. She liked the quiet — the way the barista stopped pretending to smile, the way the music lowered to a hum. The other customers had already left. Only their cups remained, ringed with residue. She opened her notebook. Wrote nothing. This had become a ritual. Not the writing, but the almost-writing. The pen held just above the page, the intention hovering like a breath before a confession. Tonight was different. A man sat at the corner table. He was reading a book she recognized — the one her mother used to keep on the nightstand.""",
        "label": "ai",
        "style": "ai_literary_fiction"
    },
    {
        "text": """Rain again. James pulled his collar up and walked faster. The streets were empty except for the usual ghosts — taxi drivers hunting for fares, a woman walking a dog that didn't want to be walked, a neon sign blinking above a bar that had been "closing soon" for three years. He passed the bookstore. Still open. Always open. He'd never gone inside. Tonight he did. The bell above the door made a sound that didn't belong in the modern world. The owner looked up from behind a desk piled with paperwork. "Looking for anything specific?" "Not really." "Best way to find something.\"""",
        "label": "ai",
        "style": "ai_literary_short"
    },
    {
        "text": """Five things I learned from failing: One. Failure is not the opposite of success. It is a suburb of success. You have to drive through it to get anywhere interesting. Two. Nobody remembers your failures as vividly as you do. That presentation you botched in 2019? Nobody else was thinking about it at 3 AM last Tuesday. Just you. Three. The fear of failure is almost always worse than failure itself. The anticipation is a horror movie. The actual event is usually just embarrassing and then it's over.""",
        "label": "ai",
        "style": "ai_listicle_casual"
    },
    {
        "text": """If you're looking to get into photography, the good news is that you don't need expensive equipment to get started. Your smartphone is actually a surprisingly capable camera, and there are countless free resources available online to help you learn the fundamentals of composition, lighting, and editing. Start by practicing with subjects that interest you — whether that's landscapes, portraits, street scenes, or everyday objects. The key is consistency and a willingness to experiment.""",
        "label": "ai",
        "style": "ai_casual_advice"
    },
    {
        "text": """The lighthouse keeper's journal had only three entries for the entire winter of 1924. The first, dated November 12, read simply: 'The gulls have flown inland. Wind from the north.' The second, on January 4: 'Light mechanism frozen. Replaced brass spindle by candlelight.' The third, on February 28, was written in an unsteady hand: 'The sea does not want us here, but it has agreed to wait.' When the supply cutter arrived in April, the lamp was still turning, but the keeper was nowhere to be found.""",
        "label": "ai",
        "style": "ai_claude_atmospheric_story"
    },
    {
        "text": """Quantum computing leverages the fundamental principles of superposition and entanglement to perform complex calculations at speeds unachievable by classical architectures. While classical bits exist deterministically as either 0 or 1, quantum bits (qubits) can exist in continuous linear combinations of both states simultaneously. This property allows quantum algorithms, such as Shor's algorithm for prime factorization and Grover's algorithm for unstructured database search, to achieve exponential and quadratic speedups respectively.""",
        "label": "ai",
        "style": "ai_technical_quantum"
    },
    {
        "text": """In the realm of modern web development, micro-frontends have emerged as an architectural pattern designed to decompose monolithic client-side applications into smaller, semi-independent micro-apps. By allowing individual engineering squads to independently develop, test, and deploy specific vertical slices of a user interface, organizations can reduce coordination overhead and foster domain-driven ownership across distributed teams.""",
        "label": "ai",
        "style": "ai_tech_architecture"
    }
]
