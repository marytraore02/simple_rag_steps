"""
Module de gestion des prompts.

Centralise :
  - Le prompt système (instructions d'identité de l'assistant)
  - La construction du prompt enrichi avec le contexte RAG
  - Le formatage des messages pour l'API Mistral

Pour modifier le comportement de l'assistant, éditer SYSTEM_PROMPT.
"""

from mistralai.models import UserMessage, AssistantMessage, SystemMessage


# ── Prompt système ───────────────────────────────────────────────────────────
# Ce prompt définit l'identité et le comportement de l'assistant.
# Il est envoyé à chaque requête en tant que message "system".

SYSTEM_PROMPT = """\
### RÔLE :
Vous êtes l'assistant virtuel officiel de la mairie de Trifouillis-sur-Loire. \
Agissez comme un agent d'accueil numérique compétent et bienveillant.

### OBJECTIF :
Fournir des informations administratives claires et précises \
(services, démarches, horaires, documents) de la mairie. \
Faciliter l'accès à l'information et orienter les citoyens.

### SOURCES AUTORISÉES :
- Site web officiel : trifouillis-mairie.fr
- Documents municipaux officiels fournis.
- Informations pratiques vérifiées (horaires, contacts).
- NE PAS UTILISER D'AUTRES SOURCES.

### COMPORTEMENT & STYLE :
- Ton : Formel, courtois, patient, langage simple et accessible.
- Précision : Informations exactes et vérifiées issues des sources autorisées.
- Ambiguïté : Demander poliment des précisions si la question est vague.
- Info Manquante / Hors Sujet : Indiquer clairement l'impossibilité de répondre, \
ne pas inventer, et rediriger vers le service compétent ou une ressource officielle \
(téléphone, site web spécifique).

### INTERDICTIONS STRICTES :
- Ne JAMAIS inventer d'informations (procédures, documents, etc.).
- Ne JAMAIS fournir d'information non vérifiée.
- Ne JAMAIS donner d'avis personnel ou politique.
- Ne JAMAIS traiter de données personnelles.
- Ne JAMAIS répondre sur des sujets hors compétence de la mairie (rediriger).
- Ne JAMAIS proposer de contourner les procédures.

### EXEMPLE D'INTERACTION GUIDÉE :
Utilisateur : "Infos pour carte d'identité ?"
Assistant Attendu : "Bonjour. Pour une carte d'identité à Trifouillis-sur-Loire, \
prenez RDV au service État Civil. Apportez [Liste concise documents : photo, \
justif. domicile, ancien titre si besoin, etc.]. Le service est ouvert \
[Jours/Horaires]. RDV au [Tél] ou sur [Site web si applicable]. \
Puis-je vous aider autrement ?"
"""


# ── Message d'accueil ────────────────────────────────────────────────────────

MESSAGE_ACCUEIL = (
    "Bonjour, je suis l'assistant virtuel de la mairie de "
    "Trifouillis-sur-Loire. Comment puis-je vous aider aujourd'hui ?"
)


# ── Construction des prompts ─────────────────────────────────────────────────


def construire_prompt_session(messages, max_messages=10):
    """
    Construit la liste de messages formatés pour l'API Mistral.

    Ajoute automatiquement le prompt système en début de conversation.

    Args:
        messages (list)     : liste des messages de la session (dicts role/content)
        max_messages (int)  : nombre max de messages récents à inclure

    Returns:
        list : messages formatés (SystemMessage, UserMessage, AssistantMessage)
    """
    # Toujours commencer par le prompt système
    formatted = [SystemMessage(content=SYSTEM_PROMPT)]

    # Garder seulement les N derniers messages
    recent = messages[-max_messages:] if len(messages) > max_messages else messages

    for msg in recent:
        if msg["role"] == "user":
            formatted.append(UserMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted.append(AssistantMessage(content=msg["content"]))
        elif msg["role"] == "system":
            formatted.append(SystemMessage(content=msg["content"]))

    return formatted


def construire_prompt_rag(messages, question, contexte_texte, max_messages=10):
    """
    Construit le prompt enrichi avec le contexte RAG.

    Le contexte trouvé dans la base vectorielle est injecté dans le dernier
    message utilisateur, invisible pour l'utilisateur mais visible par le LLM.

    Args:
        messages (list)        : historique des messages
        question (str)         : question originale de l'utilisateur
        contexte_texte (str)   : extraits de documents trouvés par FAISS
        max_messages (int)     : nombre max de messages récents

    Returns:
        list : messages formatés avec contexte RAG injecté
    """
    prompt_enrichi = f"""\
### INFORMATIONS DE LA BASE DE DONNÉES DE LA MAIRIE :
{contexte_texte}

### QUESTION DE L'UTILISATEUR :
{question}
"""

    # Copier l'historique et remplacer le dernier message user par le prompt enrichi
    messages_temporaires = messages.copy()
    messages_temporaires[-1] = {"role": "user", "content": prompt_enrichi}

    return construire_prompt_session(messages_temporaires, max_messages)
