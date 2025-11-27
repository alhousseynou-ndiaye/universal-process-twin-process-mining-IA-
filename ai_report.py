import os
import json
from typing import Dict, Any

from dotenv import load_dotenv
from groq import Groq

# Charger les variables d'environnement depuis .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY est manquant. Ajoute-le dans un fichier .env.")

client = Groq(api_key=GROQ_API_KEY)


def generate_ai_report(stats: Dict[str, Any], graph: Dict[str, Any]) -> str:
    """
    Utilise Groq pour générer un rapport d'analyse de processus
    à partir des stats et du graphe.

    Retourne un texte en français, structuré pour un décideur business.
    """

    payload = {
        "stats": stats,
        "graph_summary": {
            "num_nodes": len(graph.get("nodes", [])),
            "num_edges": len(graph.get("edges", [])),
            "top_nodes": graph.get("nodes", [])[:10],
            "top_edges": graph.get("edges", [])[:10],
        }
    }

    system_prompt = (
        "Tu es un expert en optimisation de processus, automatisation et productivité. "
        "Tu analyses des workflows métier à partir de statistiques et de graphes de transitions. "
        "Tu expliques avec des mots simples, mais avec une vraie profondeur business. "
        "Ton rôle est d'aider un dirigeant ou responsable opérationnel à comprendre son processus "
        "et à savoir quoi améliorer ou automatiser."
    )

    user_prompt = (
        "Voici les données d'un processus (en JSON). "
        "Analyse-les et produis un rapport structuré avec les sections suivantes :\n"
        "1. Résumé exécutif (3 à 5 phrases)\n"
        "2. Points forts du processus\n"
        "3. Problèmes / risques potentiels\n"
        "4. Recommandations d'amélioration (dont des idées d'automatisation concrètes)\n"
        "5. Priorités d'action (liste courte, par ordre d'impact)\n\n"
        "Données :\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=900,
    )

    report = completion.choices[0].message.content
    return report


def generate_automation_ideas(
    stats: Dict[str, Any],
    graph: Dict[str, Any],
    kpi: Dict[str, Any],
    domain: str,
) -> str:
    """
    Utilise Groq pour générer des idées d'automatisation concrètes (Make, Zapier, CRM...)
    en fonction du process (stats + graphe + KPI) et d'un contexte métier (domain).
    """

    payload = {
        "domain": domain,
        "stats": stats,
        "graph_summary": {
            "num_nodes": len(graph.get("nodes", [])),
            "num_edges": len(graph.get("edges", [])),
            "top_nodes": graph.get("nodes", [])[:10],
            "top_edges": graph.get("edges", [])[:10],
        },
        "kpi": kpi,
    }

    system_prompt = (
        "Tu es un expert en automatisation no-code (Make, Zapier, n8n, intégrations CRM) "
        "et en optimisation de processus. "
        "Tu aides les entreprises à transformer leurs processus en scénarios d'automatisation concrets."
    )

    # ⚠️ ICI : on utilise un f-string, PAS .format()
    user_prompt = f"""
Contexte métier : {domain}

Voici les données d'un processus (statistiques, graphe, KPI temps) en JSON.
Ta mission : proposer des idées d'automatisation concrètes et immédiatement actionnables.

Contraintes :
- Réponds en français.
- Reste concret : quelles données, quel déclencheur, quel outil (Make, Zapier, CRM...) ?
- Ne dépasse pas 600 mots.

Structure attendue :
1. Résumé (2-3 phrases)
2. Top 3 automatisations à mettre en place immédiatement
   - Pour chaque : déclencheur, action, outil suggéré, impact estimé
3. 2 idées d'automatisation plus avancées (si le client veut aller plus loin)
4. Données nécessaires / prérequis (ce qu'il faut pour les mettre en place)

Données du processus :
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=900,
    )

    ideas = completion.choices[0].message.content
    return ideas

