from functools import lru_cache
import os

from wtpsplit import SaT


BP = os.path.realpath(os.path.join(os.path.realpath(__file__), ".."))


@lru_cache(maxsize=1)
def load_seg_model(model_base: str, lora_path: str, language: str):
    return SaT(model_base,
               lora_path=lora_path,
               # language=language,
               )



if __name__ == "__main__":
    m = load_seg_model(model_base="sat-3l",
                       lora_path=f"{BP}/data/neg_steps_olp_3l/en",
                       language="en"
                       )
    m.half().to("cuda")
    x = {"steps": ["We need to determine correct answer under Canadian law (Ontario). Issue: risk of loss in sale of goods, when does it pass? Under Sale of Goods Act (Ontario) (now replaced by Consumer Protection Act? But generally Sale of Goods Act).",
                   "Risk of loss passes to buyer when property passes, unless parties agree otherwise.",
                   "Here, parties agreed on purchase price, but delivery not yet occurred; buyer paid in advance.",
                   "The laptop required repair (screen replacement). When does risk pass? Typically, risk passes upon delivery unless parties agree that risk passes earlier (e.g., when seller finishes performance).",
                   "Here, they agreed that after repair, Jake would let Luke know and then Luke would pick up.",
                   "Did they intend that risk passes when Jake finishes repairs? Usually risk passes upon tender of delivery.",
                   "Tender occurs when seller is ready to deliver goods to buyer, i.e., goods are in a condition to be delivered, and buyer is notified. The seller must give reasonable notice.",
                   "Here, Jake sent text saying ready June 6. That is a tender of delivery.",
                   "So risk passes at that moment (when buyer is notified and ready to take delivery).",
                   "However, the flood happened before Luke arrived. The risk had passed to buyer at the moment of tender.",
                   "Thus Jake not required to return money.",
                   "Among answer choices, which matches? Option B says: Jake is not required to return the money because risk passed as a result of completing repairs, putting laptop in deliverable state, and notifying Luke of a precise date.",
                   "That matches the principle: risk passes upon tender of delivery (goods ready for delivery and buyer notified).",
                   "Option A says insurance claim \u2013 not correct.",
                   "Option C says risk did not pass because notice not appropriate \u2013 but notice was appropriate.",
                   "Option D says risk passes only upon possession \u2013 not correct; risk passes upon tender, not possession.",
                   "Option E says risk passed June 2 because item in possession \u2013 but not delivered or ready; not correct.",
                   "So answer B.",
                   "Provide brief explanation: Under Sale of Goods Act, risk passes upon tender of delivery; Jake completed repairs and gave precise notice; risk passed then; thus Jake not required to refund."],
         "negations": [{"type": "REFINING_NEGATION", "step": 3, "text": "delivery not yet occurred", "refines_steps": [2]}]}
    y = """We need to determine correct answer under Canadian law (Ontario). Issue: risk of loss in sale of goods, when does it pass? Under Sale of Goods Act (Ontario) (now replaced by Consumer Protection Act? But generally Sale of Goods Act). Risk of loss passes to buyer when property passes, unless parties agree otherwise. Here, parties agreed on purchase price, but delivery not yet occurred; buyer paid in advance. The laptop required repair (screen replacement). When does risk pass? Typically, risk passes upon delivery unless parties agree that risk passes earlier (e.g., when seller finishes performance). Here, they agreed that after repair, Jake would let Luke know and then Luke would pick up. Did they intend that risk passes when Jake finishes repairs? Usually risk passes upon tender of delivery. Tender occurs when seller is ready to deliver goods to buyer, i.e., goods are in a condition to be delivered, and buyer is notified. The seller must give reasonable notice. Here, Jake sent text saying ready June 6. That is a tender of delivery. So risk passes at that moment (when buyer is notified and ready to take delivery). However, the flood happened before Luke arrived. The risk had passed to buyer at the moment of tender. Thus Jake not required to return money. Among answer choices, which matches? Option B says: Jake is not required to return the money because risk passed as a result of completing repairs, putting laptop in deliverable state, and notifying Luke of a precise date. That matches the principle: risk passes upon tender of delivery (goods ready for delivery and buyer notified). Option A says insurance claim – not correct. Option C says risk did not pass because notice not appropriate – but notice was appropriate. Option D says risk passes only upon possession – not correct; risk passes upon tender, not possession. Option E says risk passed June 2 because item in possession – but not delivered or ready; not correct. So answer B.

Provide brief explanation: Under Sale of Goods Act, risk passes upon tender of delivery; Jake completed repairs and gave precise notice; risk passed then; thus Jake not required to refund.

Now produce JSON."""
    print(*m.split(y.replace("\n", " "), threshold=0.586, verbose=True), sep=f"\n\n{'-' * 80}\n\n")