"""
Estimates real-world Groq token cost per turn, using the actual SYSTEM prompt
and a representative STRUCTURED STATE + history, exactly as llm.respond()
builds them -- not a guess.
"""
from dotenv import load_dotenv
load_dotenv()
import json
import llm

state = {
    "classification": {"leading_categories": ["depression"],
                       "evidence_sufficiency": "sufficient",
                       "already_introduced": False, "has_signal": True},
    "safety": {"crisis_signal": False, "crisis_patterns": [],
              "crisis_earlier_in_conversation": False},
    "domain": {"in_domain": True},
    "context": {"situational_causes": []},
    "conversation": {"turn": 3, "evidence_tokens": 21},
    "decision": {"action": "RESOLVE", "reason": "depression cleared threshold"},
}
history = [
    {"role": "user", "content": "i feel bad"},
    {"role": "assistant", "content": "Can you tell me more?"},
    {"role": "user", "content": "like nothing matters anymore and i cant get "
                                "out of bed most mornings, havent seen anyone "
                                "in weeks"},
]

convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-8:])
parts = [f"STRUCTURED STATE\n{json.dumps(state, indent=2)}",
        f"\nCONVERSATION SO FAR\n{convo}",
        "\nCATEGORY TO MENTION (hedged, never as fact):\ndepression",
        "\nWrite your reply now. Prose only, and keep it short."]
user_msg = "\n".join(parts)

sys_chars = len(llm.SYSTEM)
user_chars = len(user_msg)
sys_tok = sys_chars // 4
user_tok = user_chars // 4
total_in = sys_tok + user_tok
out_cap = 400

print(f"SYSTEM prompt:  {sys_chars:>5} chars  ~{sys_tok:>4} tokens")
print(f"per-turn state: {user_chars:>5} chars  ~{user_tok:>4} tokens")
print(f"total input:                    ~{total_in:>4} tokens")
print(f"max output cap:                  {out_cap:>4} tokens")
print(f"worst case per call:            ~{total_in + out_cap:>4} tokens")
print()

FREE_TPM, FREE_RPM, FREE_RPD = 8000, 30, 1000
worst = total_in + out_cap
typical_out = 90          # 2-3 sentences is closer to ~60-100 tokens in practice
typical = total_in + typical_out

print(f"Free tier: {FREE_RPD} requests/day, {FREE_TPM} tokens/min, {FREE_RPM} req/min")
print()
print(f"at WORST-CASE size (~{worst} tok/call):")
print(f"  daily cap (req):   {FREE_RPD} calls/day")
print(f"  token cap (min):   {FREE_TPM // worst} calls/min sustained")
print(f"  binding limit:     {'requests/day' if FREE_RPD < (FREE_TPM//worst)*60*24 else 'tokens/min'}")
print()
print(f"at TYPICAL size (~{typical} tok/call, 2-3 sentence replies):")
print(f"  daily cap (req):   {FREE_RPD} calls/day")
print(f"  token cap (min):   {FREE_TPM // typical} calls/min sustained")
print(f"  requests/day IS the binding constraint either way -- token/min ceiling")
print(f"  is never reached before the daily request count runs out.")
print()
print(f"~3 LLM calls per conversation (greet/ASK, ASK, RESOLVE) ->")
print(f"  {FREE_RPD} // 3 = {FREE_RPD//3} conversations/day")
print(f"  {FREE_RPD} sentences-worth of REPLIES/day (1 reply = 1 request)")
