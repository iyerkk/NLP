import spacy

nlp = spacy.load('en_core_web_sm')
string1 = "RIGHT FRONT BRAKES CAUGHT FIRE AND CAUSED DAMAGE TO CALIPER, ROTOR, HUB. " \
          "UNIT CRANKS BUT WON'T FIRE, DIED IN SERVICE PLAZA."

string = string1.capitalize()
# string = string2.capitalize()
doc = nlp(string)
print("------------Token")
for token in doc:
    print(token.text)
print("------------Token Lemma")
for token in doc:
    print(token.text, token.lemma_)
print("------------Part-of_Speech [POS] Tagging")
for token in doc:
    print(f'{token.text:{15}} {token.lemma_:{15}} {token.pos_:{10}} {token.is_stop}')
print("------------Dependency Parsing")
for chunk in doc.noun_chunks:
    print(f'{chunk.text:{30}} {chunk.root.text:{15}} {chunk.root.dep_}')
print("------------Named Entity Recognition")
for ent in doc.ents:
    print(ent.text, ent.label_)
print("------------Sentence Segmentation")
for sent in doc.sents:
    print(sent)

# spacy.displacy.render(doc, style='dep')
# spacy.displacy.render(doc, style='dep', options={'compact' :True, 'distance': 100})
