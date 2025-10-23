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
                       lora_path=f"{BP}/data/neg_steps_3l/en",
                       language="en"
                       )
    m.half().to("cuda")
    print(*m.split("""Okay, let's tackle this problem step by step. I need to identify the correct sequence of reagents for synthesizing [1,1'-bi(cyclopentylidene)]-2-one starting from 1,5-dichloropentane.

First, I need to understand what the target molecule is and what the starting material is:
- Starting material: 1,5-dichloropentane
- Target molecule: [1,1'-bi(cyclopentylidene)]-2-one. Hmm, the name looks a bit complex. Let me try to draw or visualize it.

The target molecule seems to be a bicyclic structure with a keto group (C=O) at position 2 on one of the cyclopentyl rings. The prefix "bi(cyclopentylidene)" suggests that there are two cyclopentyl rings connected via their 1-positions, forming a bicyclic system.

Now, let's think about the synthesis steps required to get from 1,5-dichloropentane to this target molecule.

### Step 1: Forming Cyclopentane
The starting material is 1,5-dichloropentane. To form a cyclopentane ring, I need to convert the linear pentane derivative into a cyclic structure. This typically involves an intramolecular substitution or elimination reaction.

Looking at the options:
- Option 0 starts with Zn, ether.
- Option 1 starts with Na, ether.
- Option 2 also starts with Na, ether.
- Option 3 starts with Zn, ether.

""".replace("\n", " "), threshold=0.5828, verbose=True), sep=f"\n\n{'-' * 80}\n\n")