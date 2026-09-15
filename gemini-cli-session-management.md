# Gemini CLI — Gestion des sessions (Session Management)

> Document de référence condensé, à destination d'un agent IA, pour comprendre le fonctionnement de la gestion de sessions de Gemini CLI.
> Source officielle : https://geminicli.com/docs/cli/session-management/ (dernière mise à jour de la page : 18 juin 2026)

⚠️ **Note contextuelle** : depuis le 18 juin 2026, Gemini CLI a été remplacé par **Antigravity CLI** pour les utilisateurs des paliers gratuits et Google One. Ce document décrit le fonctionnement de Gemini CLI tel que documenté ; vérifier si le produit ciblé est toujours Gemini CLI ou sa succession.

---

## 1. Concept général

Gemini CLI enregistre automatiquement l'historique de conversation, ce qui permet de reprendre le travail là où il a été laissé, de consulter les interactions passées, de gérer l'historique par projet, et de configurer la durée de rétention des données.

---

## 2. Sauvegarde automatique

L'historique de session est enregistré automatiquement en arrière-plan pendant l'utilisation, garantissant la préservation du travail même en cas d'interruption.

### Ce qui est sauvegardé
- Les prompts de l'utilisateur et les réponses du modèle.
- Toutes les exécutions d'outils (entrées et sorties).
- Les statistiques d'utilisation de tokens (input, output, cache, etc.).
- Les réflexions et résumés de raisonnement de l'assistant (quand disponibles).

### Emplacement
```
~/.gemini/tmp/<project_hash>/chats/
```
où `<project_hash>` est un identifiant unique basé sur le répertoire racine du projet.

### Portée
Les sessions sont **spécifiques à chaque projet**. Changer de répertoire de travail bascule automatiquement vers l'historique de sessions du projet correspondant.

---

## 3. Reprise de sessions

### Depuis la ligne de commande

| Commande | Effet |
|---|---|
| `gemini --resume` (ou `-r`) | Charge immédiatement la session la plus récente |
| `gemini --resume 1` | Reprend par index (voir liste via `--list-sessions`) |
| `gemini --resume a1b2c3d4-e5f6-7890-abcd-ef1234567890` | Reprend via l'UUID complet de la session |

### Depuis l'interface interactive — `/resume`

```
/resume
```
Ouvre le **Session Browser** (navigateur de sessions interactif), qui permet :
- **Browse** : parcourir la liste des sessions passées.
- **Preview** : voir la date, le nombre de messages, et le premier prompt utilisateur.
- **Search** : appuyer sur `/` pour filtrer par ID ou par contenu.
- **Select** : appuyer sur **Entrée** pour reprendre la session sélectionnée.
- **Esc** : quitter le navigateur.

Note : `/chat` est un alias fonctionnel de `/resume`. Les préfixes uniques comme `/resum` ou `/cha` résolvent vers le même menu groupé (sections `-- auto --` pour le navigateur, `-- checkpoints --` pour les checkpoints manuels nommés).

### Checkpoints manuels de conversation (points de branchement nommés)

```
/resume save decision-point
/resume list
/resume resume decision-point
```
Alias de compatibilité : `/chat ...` et `/resume checkpoints ...` (maintenu pendant la migration).

---

## 4. Sessions parallèles avec Git worktrees

Pour travailler sur plusieurs tâches simultanément, on peut utiliser les **Git worktrees** afin que chaque session Gemini dispose de sa propre copie du code, évitant les collisions entre sessions concurrentes.
→ Voir : https://geminicli.com/docs/cli/git-worktrees

---

## 5. Gestion des sessions

### Lister les sessions
```
gemini --list-sessions
```
Exemple de sortie :
```
Available sessions for this project (3):

  1. Fix bug in auth (2 days ago) [a1b2c3d4]
  2. Refactor database schema (5 hours ago) [e5f67890]
  3. Update documentation (Just now) [abcd1234]
```

### Supprimer une session

**Via ligne de commande :**
```
gemini --delete-session 2
```
(index ou ID)

**Via le Session Browser :**
1. Ouvrir avec `/resume`.
2. Naviguer jusqu'à la session à supprimer.
3. Appuyer sur **x**.

---

## 6. Configuration (`settings.json`)

### Rétention des sessions

Par défaut, Gemini CLI nettoie automatiquement les anciennes sessions pour éviter une croissance illimitée de l'historique. La suppression d'une session entraîne aussi celle de toutes les données associées : plans d'implémentation, suivis de tâches, sorties d'outils, journaux d'activité.

**Politique par défaut : rétention de 30 jours.**

```json
{
  "general": {
    "sessionRetention": {
      "enabled": true,
      "maxAge": "30d",
      "maxCount": 50
    }
  }
}
```

| Paramètre | Type | Description | Défaut |
|---|---|---|---|
| `enabled` | boolean | Interrupteur principal du nettoyage automatique | `true` |
| `maxAge` | string | Durée de conservation (ex : `"24h"`, `"7d"`, `"4w"`) ; sessions plus anciennes supprimées | `"30d"` |
| `maxCount` | number | Nombre maximum de sessions conservées ; les plus anciennes au-delà sont supprimées | non défini (illimité) |
| `minRetention` | string | Période minimale de rétention (garde-fou) ; jamais supprimée avant ce délai par le nettoyage auto | `"1d"` |

### Limites de session

Permet de limiter la longueur d'une session pour éviter une fenêtre de contexte trop grande/coûteuse.

```json
{
  "model": {
    "maxSessionTurns": 100
  }
}
```

- **`maxSessionTurns`** (number) : nombre maximal de tours (échanges utilisateur/modèle) autorisés dans une session. `-1` = illimité (valeur par défaut).

**Comportement à la limite atteinte :**
- **Mode interactif** : la CLI affiche un message informatif et arrête d'envoyer des requêtes au modèle. Il faut démarrer manuellement une nouvelle session.
- **Mode non-interactif** : la CLI se termine avec une erreur.

---

## 7. Aide-mémoire (quand utiliser quoi)

| Besoin | Commande |
|---|---|
| Reprendre la dernière session | `gemini --resume` |
| Reprendre une session précise | `gemini --resume <index\|uuid>` ou `/resume` (navigateur interactif) |
| Voir toutes les sessions d'un projet | `gemini --list-sessions` |
| Supprimer une session | `gemini --delete-session <index>` ou `x` dans le Session Browser |
| Marquer un point de branchement dans la conversation | `/resume save <nom>` |
| Revenir à un point nommé | `/resume resume <nom>` |
| Lister les checkpoints nommés | `/resume list` |
| Travailler sur plusieurs tâches sans collision | Git worktrees (une session = un worktree) |
| Limiter la taille/coût d'une session | `model.maxSessionTurns` dans `settings.json` |
| Contrôler la durée de rétention des données | `general.sessionRetention` dans `settings.json` |

---

## 8. Différences clés avec Copilot CLI (`/chronicle`)

| Aspect | Gemini CLI | Copilot CLI |
|---|---|---|
| Emplacement des données | `~/.gemini/tmp/<project_hash>/chats/` | `~/.copilot/session-state/` + `session-store.db` |
| Portée des sessions | Par projet (hash du répertoire racine) | Par session individuelle, synchronisable multi-outils |
| Synchronisation cloud | Non mentionnée dans cette page (local par projet) | Activée par défaut vers le compte GitHub |
| Recherche sémantique en langage naturel | Non native sur ce point (recherche textuelle dans le Session Browser) | Oui, via requêtes en langage naturel + `/chronicle search` (mot-clé) |
| Rapports automatiques (standup, coûts, conseils) | Non documenté ici | Oui, via `/chronicle standup`, `/chronicle cost-tips`, `/chronicle tips`, `/chronicle improve` |
| Rétention automatique configurable | Oui (`maxAge`, `maxCount`, `minRetention`) | Rétention via suppression manuelle / `/session prune --older-than` |
| Limite de longueur de session | Oui (`maxSessionTurns`) | Non mentionnée dans la doc source |
| Checkpoints nommés manuels | Oui (`/resume save/list/resume`) | Checkpoints existent en interne (table `checkpoints`) mais pas de commande de nommage manuel documentée |

---

## 9. Source

- Documentation officielle Gemini CLI — Session management : https://geminicli.com/docs/cli/session-management/
- Pages liées : [Memory tool](https://geminicli.com/docs/tools/memory), [Checkpointing](https://geminicli.com/docs/cli/checkpointing), [CLI reference](https://geminicli.com/docs/cli/cli-reference), [Git worktrees](https://geminicli.com/docs/cli/git-worktrees)
- Annonce de transition vers Antigravity CLI (18 juin 2026) : https://developers.googleblog.com/an-important-update-transitioning-gemini-cli-to-antigravity-cli

---

## 10. Notes pour un agent IA

- Les sessions sont **isolées par projet** (via le hash du chemin racine) — contrairement à Copilot CLI où les sessions peuvent être synchronisées et interrogées cross-projet/cross-outil.
- Les paramètres de rétention et de limite de session se configurent tous dans `settings.json`, modifiable aussi via `/settings` en interactif.
- Toujours vérifier si l'utilisateur cible encore "Gemini CLI" au sens strict ou si son usage relève déjà d'Antigravity CLI, le produit successeur pour les paliers gratuits/Google One depuis juin 2026.
