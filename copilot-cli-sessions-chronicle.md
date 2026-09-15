# GitHub Copilot CLI — Données de session & `/chronicle`

> Document de référence condensé, à destination d'un agent IA, pour comprendre le fonctionnement des données de session de Copilot CLI et de la fonctionnalité `/chronicle`.
> Sources : documentation officielle GitHub Docs + observations techniques non-officielles (voir section finale).

---

## 1. Concept général

Chaque session Copilot CLI génère un historique complet : prompts, réponses de Copilot, outils utilisés, fichiers modifiés. Cet historique est exploitable pour :
- interroger le travail passé en langage naturel,
- reprendre une session interrompue,
- générer des rapports automatiques (standup, coûts, conseils, etc.) via `/chronicle`.

---

## 2. Stockage des données (officiel)

| Emplacement | Type | Rôle |
|---|---|---|
| `~/.copilot/session-state/` | Fichiers bruts, un dossier par session | Enregistrement complet de la session, permet la reprise (`--continue` / `--resume`) |
| `~/.copilot/session-store.db` | Base SQLite locale ("magasin de sessions") | Sous-ensemble structuré des données ; alimente `/chronicle` et les réponses aux questions sur le travail passé |
| GitHub (cloud) | Synchronisation distante | Accès multi-outils (CLI, VS Code, JetBrains, app Copilot, GitHub.com) |

- La synchronisation vers GitHub est **activée par défaut**.
- Désactivable via `"remoteExport": false` dans les paramètres CLI JSON (les données restent alors 100 % locales, interrogeables uniquement depuis la CLI).
- En entreprise (Copilot Enterprise / Business) : un admin doit activer la politique **« Stocker les sessions locales dans le cloud »** (au minimum en mode "Afficher depuis le cloud").
- Chaque utilisateur ne peut interroger que **ses propres sessions** — isolation stricte, pas d'accès aux données d'autrui, même pour les admins.

---

## 3. Structure technique observée (non-officielle, informative)

⚠️ Ces détails ne sont **pas une API publique supportée** par GitHub. Le schéma peut changer sans préavis entre versions de Copilot CLI. À utiliser uniquement pour comprendre le fonctionnement interne, jamais pour lire/modifier directement ces fichiers en production.

```
~/.copilot/
├── config.json              # Configuration principale du CLI
├── mcp-config.json          # Configuration des serveurs MCP
├── command-history-state.json
├── session-state/           # Un dossier par session
│   └── <uuid>/
│       ├── events.jsonl     # Journal d'événements (streaming), présent pour sessions actives
│       ├── workspace.yaml   # Métadonnées de session
│       └── ...              # Autres fichiers variables (checkpoints, session.db, métadonnées VS Code...)
├── history-session-state/   # Stockage legacy (anciennes sessions)
├── agents/                  # Définitions d'agents personnalisés (*.md)
├── logs/                    # Logs de debug/erreurs
└── session-store.db         # Index SQLite global, cherchable
```

### Exemple de contenu `events.jsonl`

Chaque ligne = un événement JSON horodaté :

```jsonl
{"type":"session.start","timestamp":"...","data":{"context":{"repository":"org/repo","branch":"main"}}}
{"type":"user.message","timestamp":"...","data":{"content":"..."}}
{"type":"assistant.turn_start","timestamp":"...","data":{"turnId":"..."}}
{"type":"tool.execution_start","timestamp":"...","data":{"toolName":"...","arguments":{...}}}
{"type":"tool.execution_complete","timestamp":"...","data":{"success":true}}
{"type":"session.shutdown","timestamp":"...","data":{...}}
```

Types d'événements typiques : cycle de vie de session, messages utilisateur/assistant, exécution d'outils, changements de modèle, hooks.

### Tables principales de `session-store.db` (SQLite)

| Table | Contenu |
|---|---|
| `sessions` | ID, résumé auto-généré, dépôt, branche, timestamps |
| `turns` | Chaque message utilisateur et réponse de l'assistant |
| `checkpoints` | Snapshots titrés avec aperçu et prochaines étapes |
| `session_files` | Chaque fichier touché pendant la session |
| `session_refs` | Commits, PRs, issues liés à la session |
| `search_index` | Index FTS5 (recherche plein texte) sur tout le contenu |

---

## 4. Confidentialité

- Données locales (`~/.copilot/session-state/`) : accessibles uniquement au compte utilisateur de la machine.
- Données synchronisées : stockées côté GitHub, liées au compte personnel, accessibles uniquement à l'utilisateur par défaut.
- Activer la synchro via une politique d'organisation **ne donne pas** aux admins l'accès aux données de session.
- Possibilité de **partager une session individuelle** (accès lecture seule aux utilisateurs ayant accès au dépôt) — ces sessions partagées ne sont pas indexées pour les requêtes d'autrui.
- Quand une question porte sur l'historique ou que `/chronicle` est utilisé, les données de session (prompts, contexte, réponses passées) peuvent être envoyées au modèle IA, comme pour toute interaction normale.

---

## 5. Gestion des données de session

### Suppression via commandes slash (dans une session CLI interactive)

| Commande | Effet |
|---|---|
| `/session delete` | Supprime la session active, en démarre une nouvelle |
| `/session delete SESSION-ID` | Supprime une session précise (aperçu affiché d'abord ; ajouter `--yes` pour confirmer) |
| `/session delete-all --yes` | Supprime toutes les sessions locales sauf l'active (ignore les sessions utilisées par un autre process) |
| `/session prune --older-than DAYS [--dry-run]` | Supprime les sessions plus anciennes que N jours |

- Si la session supprimée était synchronisée, `/session delete` demande si la copie distante doit aussi être supprimée (cela la retire aussi des infos/résultats `/chronicle`).
- `/session delete-all` et `/session prune` n'affectent que les données **locales** ; pour supprimer les données synchronisées correspondantes, il faut le faire manuellement sur GitHub.com.

### Suppression manuelle

- **Local** : supprimer le dossier de session dans `~/.copilot/session-state/`. Nécessite ensuite une **réindexation manuelle** (voir section 6). N'affecte pas les données déjà synchronisées.
- **Synchronisé** : depuis GitHub.com — *masquer* une session (retire des résultats de requête, réversible) ou la *supprimer* (retire de la liste, définitif). S'applique aux sessions CLI, VS Code et app Copilot.

---

## 6. Réindexation du magasin de sessions

Le magasin (`session-store.db`) se remplit **incrémentalement** pendant une session CLI (écriture régulière + à la fin de session).

Commande de réindexation (dans une session CLI interactive) :
```
/chronicle reindex
```
Elle reconstruit l'index à partir des fichiers de `session-state/` et resynchronise avec le compte.

### Quand réindexer ?
- **Anciennes sessions** jamais indexées (fichiers créés avant l'existence du magasin de sessions).
- **Migration/récupération** : fichiers de session déplacés vers une autre machine ou restaurés depuis une sauvegarde sans le fichier `session-store.db`.
- **Corruption** : le fichier `session-store.db` est endommagé ou supprimé accidentellement → récupération possible depuis les fichiers de session.
- **Arrêt inattendu** (crash, coupure de courant) : les données en mémoire non encore écrites peuvent être récupérées si elles ont été flushées sur disque avant l'arrêt.
- Après une **suppression de session** via `/session delete*`.

---

## 7. La commande slash `/chronicle`

### Sous-commandes

| Sous-commande | Usage |
|---|---|
| `/chronicle standup last 3 days` | Génère un résumé du travail récent (rapport de stand-up) |
| `/chronicle tips` | Suggère des fonctionnalités/améliorations de workflow non exploitées |
| `/chronicle improve` | Détecte les erreurs récurrentes de l'agent et génère des instructions personnalisées correctives |
| `/chronicle cost-tips` | Analyse la consommation de tokens et propose des pistes d'économie |
| `/chronicle search KEYWORD` | Recherche **littérale** par mot-clé dans le contenu des sessions (≠ recherche sémantique) |
| `/chronicle reindex` | Reconstruit le magasin de sessions depuis les fichiers disque |

- Disponible aussi dans les IDE JetBrains, via la session CLI interactive intégrée.

### Recherche en langage naturel (différent de `/chronicle search`)
On peut poser une question libre, ex. *« Ai-je travaillé sur quelque chose lié à l'API de paiement ? »* — Copilot interprète la **sémantique** de la requête et cherche dans l'historique. `/chronicle search` fait un matching de mots-clés brut, sans interprétation sémantique.

### Reprise de session
```
copilot --continue   # reprend la dernière session locale fermée
copilot --resume      # ouvre un sélecteur de session / reprend une session spécifique
```

---

## 8. Quand utiliser quoi (aide-mémoire)

| Besoin | Commande / action |
|---|---|
| Début de journée, récap rapide | `/chronicle standup last 3 days` |
| Monter en compétence régulièrement | `/chronicle tips` (chaque semaine ou deux) |
| Copilot répète la même erreur | `/chronicle improve` |
| Comprendre la consommation de tokens | `/chronicle cost-tips` |
| Chercher un sujet précis par mot-clé | `/chronicle search KEYWORD` |
| Rappel flou d'un travail passé | Question en langage naturel |
| Reprendre un travail interrompu | `copilot --continue` / `copilot --resume` |
| Sessions corrompues/manquantes après migration | `/chronicle reindex` |

---

## 9. Sources

**Officielles (GitHub Docs) :**
- À propos des données de session Copilot CLI : https://docs.github.com/fr/copilot/concepts/agents/copilot-cli/chronicle
- Utilisation des données de session : https://docs.github.com/fr/copilot/how-tos/copilot-cli/use-copilot-cli/chronicle
- Gestion des sessions des agents : https://docs.github.com/fr/copilot/how-tos/copilot-on-github/use-copilot-agents/manage-and-track-agents
- Référence des commandes CLI : https://docs.github.com/fr/copilot/reference/copilot-cli-reference/cli-command-reference
- Répertoire de configuration CLI : https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-config-dir-reference

**Non-officielles (structure technique interne, sujette à changement) :**
- Jonathan Hoyt (employé GitHub), retour d'expérience détaillé sur `~/.copilot/` : https://jonmagic.com/posts/github-copilot-session-search-and-resume-cli/
- ⚠️ L'auteur précise explicitement que `session-store.db` et les fichiers sous `session-state/` sont des **détails d'implémentation internes**, non garantis stables entre versions, et ne constituent pas une API publique supportée.

---

## 10. Notes pour un agent IA

- **Ne pas** lire ou modifier directement `session-store.db` ou les fichiers de `session-state/` en tant que méthode d'intégration : ce ne sont pas des interfaces publiques stables.
- Préférer systématiquement les commandes officielles (`/chronicle`, `/session`, requêtes en langage naturel, `copilot --resume`/`--continue`) pour toute interaction programmatique ou assistée.
- Toujours informer l'utilisateur que la synchronisation cloud est activée par défaut et peut être désactivée (`remoteExport: false`) si la confidentialité est une priorité.
- Retenir la distinction clé : `/chronicle search` = recherche par mot-clé (littérale) vs question en langage naturel = recherche sémantique.
