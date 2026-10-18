# TP « Apprendre une densité d'images normalisée : le score matching dual » (Master TRIED)

Travaux pratiques autour de l'article de Guth, Kadkhodaie et Simoncelli,
*Learning normalized image densities via dual score matching* (NeurIPS 2025, arXiv:2506.05310).
Ce dépôt est un fork du [code des auteurs](https://github.com/FlorentinGuth/DualScoreMatching), complété pour Colab.

**Énoncé du TP** : [`tp/enonce/score_matching_dual.pdf`](enonce/score_matching_dual.pdf).

Chaque notebook commence par une cellule d'installation : sur Colab, elle télécharge **seulement** les fichiers dont
le notebook a besoin (clone partiel : code, poids du réseau utilisé, résultats pré-calculés) ; en local, elle rend
seulement le code importable. Les images (validation d'ImageNet 64×64) sont téléchargées au premier usage depuis un
miroir HuggingFace (417 Mo).

## Notebooks du TP (dossier `tp/`)

Chacun a un profil CPU (données jouets ou nombre d'images réduit) et un profil GPU (réglages des auteurs), choisis
automatiquement. Durées CPU mesurées sur un ordinateur portable à 16 cœurs ; sur le CPU de Colab (2 cœurs), elles
sont nettement plus longues (non vérifié).

| Notebook | CPU | GPU | Colab |
|---|---|---|---|
| `TP_melange_gaussien` : pourquoi le score ne suffit pas (figure 1) | 6 min | 20 000 itérations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ach0000/DualScoreMatching/blob/main/tp/TP_melange_gaussien.ipynb) |
| `TP_reseau_energie` : architecture, scores, débruitage, normalisation (tableau 1) | 2 min | plus d'images | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ach0000/DualScoreMatching/blob/main/tp/TP_reseau_energie.ipynb) |
| `TP_log_probabilites` : entropie, NLL, contenu, généralisation (figures 2 à 4, tableau 2) | 5 000 images, 6 min | 50 000 images | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ach0000/DualScoreMatching/blob/main/tp/TP_log_probabilites.ipynb) |
| `TP_dimension_effective` : dimension effective (figure 5) | 6 images, 2 min | 15 images | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ach0000/DualScoreMatching/blob/main/tp/TP_dimension_effective.ipynb) |
| `TP_entrainement_dual` : entraîner score, énergie simple et énergie duale | images 16×16, 1 h | images 64×64 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ach0000/DualScoreMatching/blob/main/tp/TP_entrainement_dual.ipynb) |

## Notebooks des auteurs adaptés pour Colab (`tp/auteurs/`)

| Notebook | Figure | Colab |
|---|---|---|
| `simple_example_colab` | figure 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ach0000/DualScoreMatching/blob/main/tp/auteurs/simple_example_colab.ipynb) |
| `denoiser_vs_energy_colab` | tableau 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ach0000/DualScoreMatching/blob/main/tp/auteurs/denoiser_vs_energy_colab.ipynb) |
| `cross_entropy_colab` | tableau 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ach0000/DualScoreMatching/blob/main/tp/auteurs/cross_entropy_colab.ipynb) |
| `generalization_colab` | figure 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ach0000/DualScoreMatching/blob/main/tp/auteurs/generalization_colab.ipynb) |
| `energy_histogram_colab` | figure 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ach0000/DualScoreMatching/blob/main/tp/auteurs/energy_histogram_colab.ipynb) |
| `affine_and_sparsity_colab` | figure 4 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ach0000/DualScoreMatching/blob/main/tp/auteurs/affine_and_sparsity_colab.ipynb) |
| `dimensionality_colab` | figure 5 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ach0000/DualScoreMatching/blob/main/tp/auteurs/dimensionality_colab.ipynb) |

Les notebooks originaux des auteurs (racine du dépôt) ne sont pas modifiés : leurs versions adaptées sont dans
`tp/auteurs/`, sous le même nom complété de `_colab`. Modifications (cellules marquées « TP ») : bouton Colab, cellule
d'installation, cellule de compatibilité (`torch.load` des points de contrôle avec PyTorch ≥ 2.6, chargement sans GPU),
`.cuda()` remplacé par `.to(DEVICE)`, pas de rendu LaTeX de matplotlib (absent de Colab), données ImageNet64
téléchargées depuis un miroir, calculs réduits sur CPU, figures et résultats écrits dans `tp/resultats/auteurs/` (et
non dans `outputs/`, qui reste celui des auteurs). Le code Python des auteurs n'est pas modifié.

## Ajouts au dépôt des auteurs

- `tp/tp_outils.py` : outils communs (chargement des réseaux sur CPU ou GPU, données, scores, laplacien de
  Hutchinson, normalisation, mélange gaussien, petits entraînements, mesures, figures).
- `tp/executer_notebooks.py` : exécute des notebooks et affiche leur durée ; les figures nommées sont écrites dans
  `tp/resultats/figures/`.
- `tp/auteurs/*_colab.ipynb` : notebooks des auteurs adaptés (voir ci-dessus).
- `tp/.gitignore` et `data/.gitignore` (créé au premier téléchargement) : résultats locaux et données jamais publiés,
  sans modifier le `.gitignore` des auteurs.

**Données.** Les auteurs utilisent la version d'ImageNet64 de Chrabaszcz et al. (2017), qu'ils ne distribuent pas.
On la télécharge depuis un miroir HuggingFace de cette version (`ChocolateDave/imagenet-64`, images PNG) : les images
de validation sont les mêmes, dans le même ordre, que celles des auteurs, si bien que leurs énergies pré-calculées
(`outputs/`) et leurs indices d'images restent valables. Conditions d'utilisation d'ImageNet : usage non commercial, de
recherche ou d'enseignement.
