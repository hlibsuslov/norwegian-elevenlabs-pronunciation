"""Source-attested lexical targets for Norwegian, Ukrainian and regional English."""
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from .common import data_dir, key
from .lookup_pronunciation import Lexicon

MFA = {
    "uk-UA": ("ukrainian_mfa.dict", "https://mfa-models.readthedocs.io/en/latest/dictionary/Ukrainian/Ukrainian%20MFA%20dictionary%20v2_0_0a.html"),
    "en-GB": ("english_uk_mfa.dict", "https://mfa-models.readthedocs.io/en/latest/dictionary/English/English%20%28UK%29%20MFA%20dictionary%20v3_1_0.html"),
}


@lru_cache(maxsize=4)
def tabular(path):
    grouped = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if "\t" not in line:
                continue
            fields = line.strip().split("\t")
            phone_spec = fields[-1].split()
            if phone_spec and len(fields)>=2:
                grouped[key(fields[0])].append(tuple(phone_spec))
    return grouped


@lru_cache(maxsize=1)
def cmu():
    import cmudict

    return cmudict.dict()


@lru_cache(maxsize=2)
def accentuated(path):
    output = defaultdict(set)
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            word = line.strip().split("\t")[0]
            if "\u0301" in word:
                unaccented = unicodedata.normalize("NFC", unicodedata.normalize("NFD", word).replace("\u0301",""))
                output[key(unaccented)].add(word)
    return output


def ipa_units(text):
    text=unicodedata.normalize("NFD",text).replace("ɡ","g")
    text=re.sub(r"[ˈˌ.'\"_/\s]", "", text)
    units=[]
    for character in text:
        if units and (unicodedata.combining(character) or character in "ːʲʰʷˑ" or units[-1].endswith(("͡","͜"))):
            units[-1]+=character
        else:
            units.append(character)
    return [unicodedata.normalize("NFC",unit.replace("͡","").replace("͜","")) for unit in units]


class LexicalRouter:
    def __init__(self, dialect="e_written", ukrainian_accent=None):
        self.norwegian = None
        self.dialect = dialect
        if (data_dir()/f"{dialect}.sqlite3").is_file():
            self.norwegian = Lexicon(dialect)
        self.ukrainian_accent = Path(ukrainian_accent).resolve() if ukrainian_accent else None

    def __enter__(self):
        return self

    def __exit__(self,*args):
        if self.norwegian:
            self.norwegian.close()

    def lookup(self, word, locale):
        if locale=="nb-NO":
            rows = self.norwegian.lookup(word) if self.norwegian else []
            result = [{"phones": row["nofabet"].split(), "alphabet":"NoFAbet", "ipa":row["ipa"], "source":row["source"],
                     "stress":row["primary_stress_phone_indices"],"secondary":row["secondary_stress_phone_indices"],
                     "tonemes":row["tonemes"],"wordform_ids":row["wordform_ids"]} for row in rows]
            return result + [{**row, "phones":ipa_units(row["ipa"]), "alphabet":"IPA",
                              "stress":None, "secondary":None} for row in result]
        if locale=="en-US":
            return [{"phones":phones, "alphabet":"CMU", "ipa":None,
                     "source":"https://github.com/cmusphinx/cmudict",
                     "stress":[i for i,p in enumerate(phones) if p.endswith("1")],
                     "secondary":[i for i,p in enumerate(phones) if p.endswith("2")]}
                    for phones in cmu().get(key(word),[])]
        if locale in MFA:
            filename, source = MFA[locale]
            file = data_dir()/"multilingual"/filename
            if not file.is_file():
                return []
            stresses = sorted(accentuated(str(self.ukrainian_accent)).get(key(word),[])) if locale=="uk-UA" and self.ukrainian_accent else []
            candidates = tabular(str(file)).get(key(word),[])
            result = [{"phones":list(phones),"alphabet":"MFA_IPA", "ipa":" ".join(phones),"source":source,
                     "accented_spellings":stresses,"stress":None,"needs_normative_stress_review":not bool(stresses)}
                    for phones in dict.fromkeys(candidates)]
            return result + [{**row,"phones":[p for phone in row["phones"] for p in ipa_units(phone)],
                              "alphabet":"IPA"} for row in result]
        return []
