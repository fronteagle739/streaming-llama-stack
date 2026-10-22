# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/gguf_qat/sidecar_dataset.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import random
from typing import Dict, List, Tuple
FACTS: List[Tuple[str, str, List[str], str, List[str]]] = [('eurovision', 'culture', ['Who won the Eurovision Song Contest 2026?', 'Which country won Eurovision 2026?', 'Name the winner of the 2026 Eurovision Song Contest.', 'Who took first place at Eurovision 2026?'], 'Bulgaria won the Eurovision Song Contest 2026, with Dara performing the song "Bangaranga" in Vienna.', ['Bulgaria', 'Dara', 'Bangaranga']), ('indy500', 'sports', ['Who won the 2026 Indianapolis 500?', 'Name the winner of the Indianapolis 500 in 2026.', 'Which driver won the Indy 500 in 2026?', 'Who took the checkered flag at the 2026 Indy 500?'], 'Felix Rosenqvist of Sweden won the 2026 Indianapolis 500, edging David Malukas by 0.0233 seconds in the closest finish in race history.', ['Rosenqvist', 'Felix Rosenqvist']), ('hungary', 'politics', ['Who became Prime Minister of Hungary in May 2026?', "Who defeated Viktor Orban in Hungary's 2026 election?", "Who succeeded Viktor Orban as Hungary's leader in 2026?", "Name Hungary's new prime minister after the May 2026 election."], 'Peter Magyar succeeded Viktor Orban as Prime Minister of Hungary on May 9, 2026, after Orban was defeated in the general election.', ['Magyar', 'Peter Magyar']), ('opec', 'politics', ['Which country left OPEC in May 2026?', 'Name the nation that exited OPEC in 2026 after six decades.', 'Who withdrew from OPEC in May 2026?', 'Which oil producer quit OPEC in 2026?'], 'The United Arab Emirates left OPEC in May 2026, ending six decades of membership.', ['United Arab Emirates', 'UAE', 'Emirates']), ('ebola', 'health', ['In which countries did the WHO declare an Ebola public health emergency in May 2026?', 'Where was the 2026 Ebola outbreak that the WHO declared an emergency over?', 'Name the countries hit by the Ebola outbreak the WHO flagged in May 2026.', 'The WHO declared a 2026 Ebola emergency in which African nations?'], 'The WHO declared a public health emergency over an Ebola outbreak in the Democratic Republic of the Congo and Uganda in May 2026.', ['Congo', 'Uganda', 'Democratic Republic']), ('kentucky_derby', 'sports', ['Which horse won the 2026 Kentucky Derby?', 'Name the winner of the 152nd Kentucky Derby.', 'Who won the Kentucky Derby in May 2026?', 'What horse took the 2026 Kentucky Derby?'], 'Golden Tempo won the 152nd Kentucky Derby in 2026 at 24-1 odds, with Cherie DeVaux becoming the first female trainer to win the race.', ['Golden Tempo']), ('derby_trainer', 'sports', ['Who was the first female trainer to win the Kentucky Derby?', 'Name the trainer of the 2026 Kentucky Derby winner.', 'Which trainer made history at the 2026 Kentucky Derby?', 'Who trained the 152nd Kentucky Derby winner?'], 'Cherie DeVaux trained Golden Tempo to win the 2026 Kentucky Derby, becoming the first female trainer to win the race.', ['DeVaux', 'Cherie DeVaux']), ('polish_cup', 'sports', ['Who won the 2026 Polish Cup final?', 'Name the winner of the Polish Cup in 2026.', 'Which club won the Polish Cup final in May 2026?', 'Who lifted the Polish Cup in 2026?'], 'Gornik Zabrze won the 2026 Polish Cup, defeating Rakow Czestochowa 2-0 for their 7th title.', ['Gornik', 'Gornik Zabrze', 'Zabrze']), ('antigua', 'politics', ['Who won the 2026 general election in Antigua and Barbuda?', "Which party won Antigua and Barbuda's 2026 election?", "Who was sworn in for a fourth term as Antigua and Barbuda's leader in 2026?", "Name the winner of Antigua and Barbuda's May 2026 election."], "Gaston Browne's ABLP won 15 of 17 seats in Antigua and Barbuda's 2026 general election, and Browne was sworn in for a fourth term.", ['Browne', 'Gaston Browne', 'ABLP']), ('romania', 'politics', ['Which Romanian prime minister lost a no-confidence vote in May 2026?', 'Who was the Romanian PM ousted by a no-confidence motion in 2026?', 'Name the Romanian prime minister whose government collapsed in May 2026.', 'Against which Romanian PM did Parliament pass a no-confidence motion in 2026?'], "Romania's Parliament voted 281-4 to pass a no-confidence motion against Prime Minister Ilie Bolojan in May 2026, collapsing the pro-EU coalition.", ['Bolojan', 'Ilie Bolojan']), ('canada_gg', 'politics', ["Who was appointed Canada's next governor general in May 2026?", 'Name the person appointed governor general of Canada in 2026.', "Who did PM Mark Carney appoint as Canada's governor general in 2026?", "Who became Canada's governor general designate in May 2026?"], "Louise Arbour was appointed Canada's next governor general in May 2026 by Prime Minister Mark Carney.", ['Arbour', 'Louise Arbour']), ('florida_openai', 'tech', ['Which U.S. state sued OpenAI CEO Sam Altman in June 2026?', 'Name the state that filed a lawsuit against ChatGPT and Sam Altman in 2026.', "Who sued OpenAI's Sam Altman over ChatGPT safety in June 2026?", 'Which state accused ChatGPT of putting profit over safety in 2026?'], 'The U.S. state of Florida filed a lawsuit against OpenAI CEO Sam Altman and ChatGPT in June 2026, accusing the chatbot of putting profit over safety.', ['Florida']), ('worldcup', 'sports', ['When did the 2026 FIFA World Cup begin?', 'What date did the FIFA World Cup 2026 start?', 'On which day did the 2026 World Cup kick off?', 'When does the 2026 FIFA World Cup open?'], 'The 2026 FIFA World Cup began on June 11, 2026, hosted across 16 cities in the United States, Mexico, and Canada.', ['June 11']), ('shanxi_mine', 'disaster', ['How many people died in the Liushenyu coal mine explosion in 2026?', 'What was the death toll of the Shanxi coal mine gas explosion in May 2026?', 'How many were killed in the 2026 Liushenyu mine blast?', 'Give the death toll of the Shanxi, China coal mine explosion in 2026.'], 'A gas explosion at the Liushenyu coal mine in Shanxi, China killed 82 people in May 2026.', ['82']), ('french_open', 'sports', ['Who became the lowest-ranked French Open finalist in history in 2026?', 'Name the No. 114 ranked player who reached the 2026 French Open final.', 'Which player was the lowest-ranked finalist at the 2026 French Open?', 'Who upset Diana Shnaider to reach the 2026 French Open final?'], 'Maja Chwalinska, ranked No. 114, became the lowest-ranked finalist in French Open history in June 2026, defeating Diana Shnaider.', ['Chwalinska', 'Maja Chwalinska'])]
HELDOUT_FACT_IDS = {'french_open', 'shanxi_mine', 'canada_gg'}

def _qa_to_chat_pairs(fact, only_first_phrasing_for_train: bool=False):
    fid, cat, qs, ans, keys = fact
    use = qs[:1] if only_first_phrasing_for_train else qs
    return [(q, ans) for q in use]

def build_train_pairs(seed: int=0, paraphrase: bool=True, mix_neutral: int=0) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    pairs: List[Tuple[str, str]] = []
    for f in FACTS:
        if f[0] in HELDOUT_FACT_IDS:
            continue
        pairs.extend(_qa_to_chat_pairs(f, only_first_phrasing_for_train=not paraphrase))
    if mix_neutral:
        from cosmos_windows.gguf_qat.train_gguf_lora_gemma4 import NEUTRAL
        for _ in range(mix_neutral):
            pairs.append(rng.choice(NEUTRAL))
    rng.shuffle(pairs)
    return pairs

def build_eval_facts(split: str='all'):
    out = []
    for fid, cat, qs, ans, keys in FACTS:
        is_held = fid in HELDOUT_FACT_IDS
        if split == 'heldout' and (not is_held):
            continue
        if split == 'trained' and is_held:
            continue
        out.append((fid, cat, qs[-1], ans, keys, is_held))
    return out

def score_answer(generated: str, keys: List[str]) -> bool:
    g = generated.lower()
    return any((k.lower() in g for k in keys))

def summary() -> Dict[str, object]:
    cats: Dict[str, int] = {}
    for _, cat, *_ in FACTS:
        cats[cat] = cats.get(cat, 0) + 1
    train = build_train_pairs(paraphrase=True)
    return {'n_facts': len(FACTS), 'categories': cats, 'heldout_facts': sorted(HELDOUT_FACT_IDS), 'n_train_facts': len(FACTS) - len(HELDOUT_FACT_IDS), 'n_train_pairs_with_paraphrase': len(train)}
if __name__ == '__main__':
    import json
    s = summary()
    print(json.dumps(s, indent=2))
    print('\n--- sample train pairs ---')
    for q, a in build_train_pairs(paraphrase=True)[:4]:
        print(f'Q: {q}\nA: {a}\n')
    print('--- eval facts (heldout) ---')
    for fid, cat, q, a, keys, held in build_eval_facts('heldout'):
        print(f'[{cat}] {fid}: {q}  -> keys={keys}')
