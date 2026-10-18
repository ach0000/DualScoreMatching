'''
Outils communs aux notebooks du TP « Densités d'images normalisées par score matching dual »
(Guth, Kadkhodaie, Simoncelli, NeurIPS 2025, arXiv:2506.05310).

Ce module complète le code des auteurs (répertoire parent) sans le modifier :
 - chargement sur CPU ou GPU des trois réseaux fournis (énergie niveaux de gris, énergie couleur,
   débruiteur couleur), sans ImageNet64 ni DataParallel ;
 - données : ImageNet64 (version de Chrabaszcz et al., celle des auteurs), d'après un miroir HuggingFace ;
 - énergie, scores d'espace et de temps, laplacien (estimateur de Hutchinson), constante de normalisation ;
 - mélange de deux gaussiennes de l'article (figure 1) : énergie exacte, petit modèle d'énergie, entraînement
   par score matching simple ou dual ;
 - mesures (PSNR, unités logarithmiques, ajustement de Gumbel, rapport de participation) et affichages.

Conventions des auteurs reprises ici : images dans [0, 1], de forme (B, C, 64, 64) ; t désigne la VARIANCE du
bruit par pixel (y = x + sqrt(t) z) ; U(y, t) est l'énergie, c'est-à-dire -log p(y | t), en nats ;
∇_y U est le score d'espace changé de signe, ∂_t U le score de temps changé de signe.
'''
import io
import math
import os
import pickle
import sys
import time
import types
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import noise  # noqa: E402  (code des auteurs)
from noise import NoiseLevel, SignalShape, IdentityCovariance, UniformLog, noisy_batch  # noqa: E402
from networks.model import EnergyModel, DenoiserModel, Reparameterization, ModelInput, LogTensor  # noqa: E402
from networks.unet import UNet  # noqa: E402

# Réseaux fournis par les auteurs (dossier models/ du dépôt)
MODELES = {
    'gris': 'finalclean_imagenet64_grayscale_lr0.0005_1Msteps10decays',
    'couleur': 'finalclean_imagenet64_color_lr0.0005_1Msteps10decays',
    'debruiteur_couleur': 'finalclean_imagenet64_color_denoiser_psco_lr0.0005_1Msteps10decays',
}
T_MIN, T_MAX = 1e-9, 1e3   # plage des variances de bruit de l'entraînement (PSNR d'entrée de 90 à -30 dB)


def fixer_graine(graine):
    np.random.seed(graine)
    torch.manual_seed(graine)


# ------------------------------------------------------------------------------------------------
# Cache des calculs longs (reprise après interruption)
# ------------------------------------------------------------------------------------------------
DOSSIER_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultats', 'cache')


def empreinte(X):
    '''Résumé d'un tenseur (forme et sommes), pour que le cache ne serve qu'aux mêmes données.'''
    X = torch.as_tensor(X)
    plat = X.flatten().double()
    return (tuple(X.shape), round(plat.sum().item(), 6), round(plat[::997].sum().item(), 6))


def chemin_cache(nom, **parametres):
    '''Fichier de cache propre à un nom et à des paramètres (tout changement de paramètre donne un autre fichier).'''
    import hashlib
    cle = hashlib.md5(repr(sorted(parametres.items())).encode()).hexdigest()[:10]
    os.makedirs(DOSSIER_CACHE, exist_ok=True)
    return os.path.join(DOSSIER_CACHE, '%s_%s.pt' % (nom, cle))


def sauver(objet, chemin):
    '''Écriture atomique : une interruption pendant l'écriture ne laisse pas de fichier corrompu.'''
    torch.save(objet, chemin + '.part')
    os.replace(chemin + '.part', chemin)


def charger(chemin):
    return torch.load(chemin, map_location='cpu', weights_only=False)


def par_morceaux(chemin, n, taille, calcul):
    '''Calcule calcul(debut, fin) (un tenseur) par morceaux de taille images, chaque morceau étant enregistré :
    une exécution interrompue reprend au premier morceau manquant. Renvoie la concaténation.'''
    resultats = []
    for debut in range(0, n, taille):
        fin = min(n, debut + taille)
        morceau = chemin.replace('.pt', '_%07d_%07d.pt' % (debut, fin))
        if os.path.exists(morceau):
            resultats.append(charger(morceau))
        else:
            r = calcul(debut, fin)
            sauver(r, morceau)
            resultats.append(r)
    return torch.cat(resultats)


def mode_images():
    '''Le code des auteurs lit la forme des signaux dans une variable globale (noise.s) : 3 dimensions (C, H, W)
    pour des images, 1 pour des vecteurs (mélange gaussien). À rappeler si l'on passe de l'un à l'autre.'''
    noise.s = SignalShape(3)


def mode_vecteurs():
    noise.s = SignalShape(1)


# ------------------------------------------------------------------------------------------------
# Unités
# ------------------------------------------------------------------------------------------------
def nats_vers_dB(e, d):
    '''Énergie totale (nats) -> log-probabilité en décibels par dimension, 10 log10 p / d.'''
    return -10 * e / (d * math.log(10))


def nats_vers_bits(e, d):
    '''Énergie totale (nats) -> énergie en bits par dimension (densité continue, sans quantification).'''
    return e / (d * math.log(2))


def nll_discrete_bits(e, d, niveaux=256):
    '''NLL d'un modèle discret obtenu en supposant la densité constante dans chaque case de quantification
    (largeur 1/niveaux) : on ajoute log2(niveaux) = 8 bits par dimension.'''
    return nats_vers_bits(e, d) + math.log2(niveaux)


def psnr(x, x_chapeau):
    '''PSNR (dB) par image, pour des images dans [0, 1] : -10 log10(MSE).'''
    mse = ((x - x_chapeau) ** 2).flatten(1).mean(1)
    return -10 * torch.log10(mse)


# ------------------------------------------------------------------------------------------------
# Chargement des réseaux des auteurs
# ------------------------------------------------------------------------------------------------
class _Depickleur(pickle.Unpickler):
    '''Les points de contrôle ont été écrits depuis main.py exécuté comme script : leurs classes y sont
    rattachées au module __main__, qu'on redirige vers le module main des auteurs.'''
    def find_class(self, module, nom):
        if module == '__main__':
            module = 'main'
        return super().find_class(module, nom)


_PICKLE = types.SimpleNamespace(Unpickler=_Depickleur, load=pickle.load, __name__='pickle')


def _sans_data_parallel(modele):
    '''Avec un GPU, la classe Model des auteurs enveloppe le réseau dans DataParallel ; comme les auteurs dans leurs
    notebooks (load_exp), on retire cette enveloppe (un seul GPU, accès direct au réseau et à ses poids).'''
    if isinstance(modele.network, nn.DataParallel):
        modele.network = modele.network.module
    return modele


class Modele:
    '''Réseau des auteurs prêt à l'emploi.

    Attributs : reseau (EnergyModel ou DenoiserModel des auteurs), info (statistiques du jeu : taille, canaux,
    moyenne, variance), d (dimension), constante (constante de normalisation de l'équation (9), en nats),
    perf_test (mesures enregistrées par les auteurs pendant l'entraînement), args (arguments d'entraînement).'''

    def __init__(self, nom):
        mode_images()
        self.nom = nom
        dossier = os.path.join(RACINE, 'models', MODELES.get(nom, nom))
        point = torch.load(os.path.join(dossier, 'model.pth.tar'), map_location='cpu', weights_only=False,
                           pickle_module=_PICKLE)
        self.args = point['args']
        self.perf_test = point['test_perf']
        self.info = self.perf_test.noise_levels.dataset_info
        self.d = self.info.dimension
        self.est_energie = self.args.model == 'EnergyModel'
        unet = UNet(dataset_info=self.info, **eval(self.args.network_kwargs))
        enveloppe = Reparameterization(unet, dataset_info=self.info, **eval(self.args.reparam_kwargs))
        enveloppe.load_state_dict(point['state_dict'])
        classe = EnergyModel if self.est_energie else DenoiserModel
        self.reseau = _sans_data_parallel(classe(enveloppe.to(DEVICE), self.info, IdentityCovariance()))
        self.reseau.eval()
        for p in self.reseau.parameters():
            p.requires_grad = False
        self.constante = getattr(self.perf_test, 'log_normalization_constant', None)
        if self.est_energie:
            self.reseau.log_normalization_constant = self.constante   # énergies normalisées (comme les auteurs)
        self.nb_parametres = sum(p.numel() for p in enveloppe.parameters())

    @classmethod
    def depuis_reseau(cls, enveloppe, info, est_energie, constante=None):
        '''Même interface pour un réseau entraîné dans le notebook (enveloppe : Reparameterization des auteurs).'''
        m = cls.__new__(cls)
        m.nom = None
        m.args, m.perf_test, m.info, m.d, m.est_energie = None, None, info, info.dimension, est_energie
        classe = EnergyModel if est_energie else DenoiserModel
        m.reseau = _sans_data_parallel(classe(enveloppe.to(DEVICE), info, IdentityCovariance()))
        m.constante = constante
        m.nb_parametres = sum(p.numel() for p in enveloppe.parameters())
        return m

    def fixer_constante(self, constante):
        self.constante = constante
        self.reseau.log_normalization_constant = constante

    @property
    def unet(self):
        return self.reseau.network.network

    def entree(self, y, t):
        '''ModelInput des auteurs pour un lot y (B, C, H, W) et une variance t (scalaire ou (B,)).'''
        t = torch.as_tensor(t, dtype=torch.float32, device=y.device)
        if t.ndim == 0:
            t = t.expand(y.shape[0])
        return ModelInput(noisy=y, noise_level=NoiseLevel(self.info, variance=t))

    def energie_brute(self, y, t):
        '''Sortie scalaire du réseau, U_theta(y, t) = <y, s_theta(y, t)> / 2, SANS la constante (nats).'''
        t = torch.as_tensor(t, dtype=torch.float32, device=y.device)
        if t.ndim == 0:
            t = t.expand(y.shape[0])
        return self.reseau.network(y, t)

    def energie(self, X, t=0.0, taille_lot=256, normaliser=True):
        '''Énergies U_theta(x, t) (nats, normalisées par défaut) d'un ensemble d'images, par lots, sans gradient.'''
        sorties = []
        with torch.no_grad():
            for lot in X.split(taille_lot):
                lot = lot.to(DEVICE)
                tt = torch.full((lot.shape[0],), float(t), device=DEVICE)
                y = lot if t == 0 else lot + tt.sqrt()[:, None, None, None] * torch.randn_like(lot)
                e = self.energie_brute(y, tt)
                if normaliser and self.constante is not None:
                    e = e + self.constante
                sorties.append(e.cpu())
        return torch.cat(sorties)

    def log_probabilites(self, X, taille_lot=256, cache=None, taille_morceau=1000):
        '''log p_theta(x) en dB par dimension (t = 0). Avec cache=<nom>, le calcul est fait par morceaux de
        taille_morceau images, enregistrés dans resultats/cache/ : il reprend après une interruption.'''
        if cache is None or self.nom is None:
            return nats_vers_dB(self.energie(X, 0.0, taille_lot), self.d)
        chemin = chemin_cache('logp_' + cache, modele=self.nom, donnees=empreinte(X), constante=self.constante)
        return par_morceaux(chemin, len(X), taille_morceau,
                            lambda a, b: nats_vers_dB(self.energie(X[a:b], 0.0, taille_lot), self.d))

    def scores(self, y, t):
        '''Énergie (normalisée), ∇_y U, ∂_t U et image débruitée y - t ∇_y U, calculés par le code des auteurs
        (une rétropropagation). Pour le débruiteur, seuls ∇_y U = (y - f(y)) / t et f(y) sont définis.'''
        sortie = self.reseau(self.entree(y.to(DEVICE), t), create_graph=False)
        return sortie

    def debruiter(self, Y, t, taille_lot=64):
        sorties = []
        for lot in Y.split(taille_lot):
            sorties.append(self.scores(lot, t).denoised.detach().cpu())
        return torch.cat(sorties)


def charger_modele(nom='gris'):
    '''nom : 'gris' (énergie, ImageNet64 en niveaux de gris), 'couleur' (énergie, couleur),
    'debruiteur_couleur' (réseau de score, couleur). Voir MODELES.'''
    debut = time.time()
    m = Modele(nom)
    print('%s : %s, %.2f M paramètres, chargé en %.1f s sur %s' % (
        nom, 'réseau d\'énergie' if m.est_energie else 'réseau de score (débruiteur)', m.nb_parametres / 1e6,
        time.time() - debut, DEVICE))
    return m


def derivees(m, y, t, graphe=False):
    '''Calcul explicite (sans passer par EnergyModel) : U = réseau(y, t) scalaire, puis ses dérivées
    ∇_y U et ∂_t U par autograd. Renvoie (U, ∇_y U, ∂_t U) ; graphe=True garde le graphe (dérivées secondes).'''
    y = y.to(DEVICE).detach().requires_grad_(True)
    t = torch.as_tensor(t, dtype=torch.float32, device=DEVICE)
    if t.ndim == 0:
        t = t.expand(y.shape[0])
    t = t.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        U = m.energie_brute(y, t)
        gy, gt = torch.autograd.grad(U.sum(), (y, t), create_graph=graphe)
    return U, gy, gt, y, t


def laplacien_hutchinson(m, y, t, n_sondes=8, graine=0):
    '''Estimateur de Hutchinson du laplacien ΔU = tr ∇²_y U : moyenne de v^T ∇²U v sur des sondes v de
    Rademacher (E[v v^T] = I). Une rétropropagation de plus par sonde (double rétropropagation).'''
    g = torch.Generator(device='cpu').manual_seed(graine)
    U, gy, gt, y, t = derivees(m, y, t, graphe=True)
    estimations = []
    for _ in range(n_sondes):
        v = (torch.randint(0, 2, y.shape, generator=g) * 2 - 1).float().to(DEVICE)
        hv = torch.autograd.grad((gy * v).sum(), y, retain_graph=True)[0]
        estimations.append((hv * v).flatten(1).sum(1).detach())
    lap = torch.stack(estimations)
    return U.detach(), gy.detach(), gt.detach(), lap.mean(0), lap.std(0) / math.sqrt(n_sondes)


def constante_normalisation(m, X, t_max=T_MAX, taille_lot=256, graine=0):
    '''Équation (9) : C = d/2 log(2 π e t_max) - E_y[U_theta(y, t_max)], y = x + sqrt(t_max) z.'''
    fixer_graine(graine)
    e = m.energie(X, t_max, taille_lot, normaliser=False)
    return m.d / 2 * math.log(2 * math.pi * math.e * t_max) - e.mean().item(), e


# ------------------------------------------------------------------------------------------------
# Données : ImageNet64 (version de Chrabaszcz et al., 2017, celle des auteurs)
# ------------------------------------------------------------------------------------------------
# Miroir HuggingFace de la version de Chrabaszcz et al. (images PNG, sans perte) : mêmes images de validation, dans le
# même ordre, que celles des auteurs (vérifié : log-probabilités identiques à celles de outputs/energies_test.pt aux
# retournements horizontaux près), ce qui rend utilisables leurs énergies pré-calculées et leurs indices d'images.
MIROIR = 'https://huggingface.co/datasets/ChocolateDave/imagenet-64/resolve/main/data/'
URL_IMAGENET64_VAL = MIROIR + 'val-00000-of-00001.parquet'                 # 50 000 images, 417 Mo
URL_IMAGENET64_TRAIN = MIROIR + 'train-{:05d}-of-00022.parquet'           # 22 fragments d'environ 58 000 images
DOSSIER_DONNEES = os.path.join(RACINE, 'data', 'imagenet64_chrabaszcz')


def _dossier_donnees(dossier):
    '''Crée un dossier de data/ (racine du fork) et y place un .gitignore qui ignore tout : les données téléchargées ne
    sont jamais publiées, sans modifier le .gitignore du dépôt des auteurs.'''
    os.makedirs(dossier, exist_ok=True)
    garde = os.path.join(RACINE, 'data', '.gitignore')
    if not os.path.exists(garde):
        with open(garde, 'w') as f:
            f.write('# données téléchargées par le TP (tp/tp_outils.py) : jamais publiées\n*\n')


def _telecharger(url, chemin):
    '''Téléchargement reprenable : un fichier partiel (.part) est complété à la reprise (en-tête Range).'''
    if os.path.exists(chemin):
        return
    _dossier_donnees(os.path.dirname(chemin))
    partiel = chemin + '.part'
    for essai in range(20):
        deja = os.path.getsize(partiel) if os.path.exists(partiel) else 0
        requete = urllib.request.Request(url, headers={'Range': 'bytes=%d-' % deja} if deja else {})
        try:
            with urllib.request.urlopen(requete, timeout=60) as reponse:
                if deja and reponse.status != 206:     # le serveur ignore Range : on repart de zéro
                    deja = 0
                total = deja + int(reponse.headers.get('Content-Length', 0))
                print('téléchargement de %s (%.0f Mo%s)' % (url, total / 1e6, ', reprise' if deja else ''),
                      flush=True)
                with open(partiel, 'ab' if deja else 'wb') as f:
                    while True:
                        bloc = reponse.read(1 << 20)
                        if not bloc:
                            break
                        f.write(bloc)
            os.replace(partiel, chemin)
            return
        except (OSError, urllib.error.URLError) as erreur:
            print('  interruption du téléchargement (%s) : nouvel essai' % erreur, flush=True)
            time.sleep(5)
    raise RuntimeError('téléchargement impossible : ' + url)


def _decoder_parquet(chemin, indices=None):
    '''Images RGB uint8 (N, 64, 64, 3) et classes d'un fichier parquet du miroir.'''
    from PIL import Image
    import pyarrow.parquet as pq
    table = pq.read_table(chemin)
    if indices is not None:
        table = table.take(indices)
    images = table.column('image').to_pylist()
    couleur = np.stack([np.asarray(Image.open(io.BytesIO(im['bytes'])).convert('RGB')) for im in images])
    return couleur, np.asarray(table.column('label').to_pylist())


def preparer_imagenet64_val():
    '''Télécharge (417 Mo) et décode les 50 000 images de validation d'ImageNet64, puis les enregistre en uint8 :
    couleur (N, 64, 64, 3) et niveaux de gris (N, 64, 64), conversion PIL « L » comme torchvision.transforms.Grayscale
    chez les auteurs. Quelques minutes ; reprenable.'''
    from PIL import Image
    if all(os.path.exists(os.path.join(DOSSIER_DONNEES, n)) for n in ('val_couleur.npy', 'val_gris.npy', 'val_classes.npy')):
        print('images déjà préparées dans %s' % DOSSIER_DONNEES)
        return
    parquet = os.path.join(DOSSIER_DONNEES, 'validation.parquet')
    _telecharger(URL_IMAGENET64_VAL, parquet)
    couleur, classes = _decoder_parquet(parquet)
    gris = np.stack([np.asarray(Image.fromarray(c).convert('L')) for c in couleur])
    for nom, tableau in [('val_couleur.npy', couleur), ('val_gris.npy', gris), ('val_classes.npy', classes)]:
        np.save(os.path.join(DOSSIER_DONNEES, nom + '.part.npy'), tableau)
        os.replace(os.path.join(DOSSIER_DONNEES, nom + '.part.npy'), os.path.join(DOSSIER_DONNEES, nom))
    print('%d images enregistrées dans %s' % (len(couleur), DOSSIER_DONNEES))


# Statistiques d'ImageNet64 enregistrées dans les points de contrôle des auteurs (test_perf...dataset_info)
STATISTIQUES_IMAGENET64 = {True: (0.4568142592906952, 0.06533287465572357),
                           False: (0.4476744532585144, 0.07243767380714417)}


def preparer_imagenet64_auteurs():
    '''Écrit data/imagenet64/ au format qu'attend le code des auteurs (classe ImageNet64 de data.py, format
    « pickle » de Chrabaszcz et al. : dictionnaires {'data': (N, 3*64*64) uint8 en ordre C, H, W, 'labels': [...]}),
    ainsi que les fichiers de statistiques (info_*.pt) que data.load_data mémorise : ils sont repris des points de
    contrôle des auteurs, ce qui évite de les recalculer (le calcul des auteurs exige un GPU).
      - val_data : les 50 000 images de validation, dans l'ordre des auteurs ;
      - train_data_batch_1 : le premier des 22 fragments d'entraînement du miroir (environ 58 000 images, 483 Mo) ;
        train_data_batch_2 à 10 sont vides. Suffisant pour les notebooks des auteurs, qui n'utilisent les images
        d'entraînement que pour comparer les NLL d'entraînement et de test.'''
    from memoize import short_name
    dossier = os.path.join(RACINE, 'data', 'imagenet64')
    _dossier_donnees(dossier)

    def ecrire(nom, X, y):
        chemin = os.path.join(dossier, nom)
        with open(chemin + '.part', 'wb') as f:
            pickle.dump({'data': X.transpose(0, 3, 1, 2).reshape(len(X), 3 * 64 * 64), 'labels': list(y)}, f)
        os.replace(chemin + '.part', chemin)

    if not os.path.exists(os.path.join(dossier, 'val_data')):
        parquet = os.path.join(DOSSIER_DONNEES, 'validation.parquet')
        _telecharger(URL_IMAGENET64_VAL, parquet)
        ecrire('val_data', *_decoder_parquet(parquet))
    if not os.path.exists(os.path.join(dossier, 'train_data_batch_10')):
        parquet = os.path.join(DOSSIER_DONNEES, 'train0.parquet')
        _telecharger(URL_IMAGENET64_TRAIN.format(0), parquet)
        ecrire('train_data_batch_1', *_decoder_parquet(parquet))
        for k in range(2, 11):
            ecrire('train_data_batch_%d' % k, np.zeros((0, 64, 64, 3), np.uint8), [])
    ecrire_infos_auteurs()
    print('ImageNet64 prêt dans', dossier)


def ecrire_infos_auteurs():
    '''Fichiers de statistiques d'ImageNet64 (data/imagenet64/info_*.pt) que data.load_data lit avant de construire un
    réseau des auteurs ; repris de leurs points de contrôle (aucun téléchargement).'''
    from memoize import short_name
    from data import DatasetInfo
    dossier = os.path.join(RACINE, 'data', 'imagenet64')
    _dossier_donnees(dossier)
    for gris, (moyenne, variance) in STATISTIQUES_IMAGENET64.items():
        chemin = os.path.join(dossier, short_name('info', spatial_size=None, grayscale=gris, horizontal_flip=True))
        if not os.path.exists(chemin):
            torch.save(DatasetInfo(spatial_size=64, num_channels=1 if gris else 3, mean=moyenne, variance=variance),
                       chemin)


def charger_imagenet64_val(n=None, gris=True, debut=0, indices=None):
    '''Images de validation d'ImageNet64 (préparées par preparer_imagenet64_val, cellule « Données » des notebooks),
    tenseur float (n, C, 64, 64) dans [0, 1] : C = 1 (gris) ou 3 (couleur) ; les n images à partir de debut, ou celles
    des indices donnés (numérotation des auteurs). Ces images n'ont servi à entraîner aucun des réseaux.'''
    fichier = os.path.join(DOSSIER_DONNEES, 'val_gris.npy' if gris else 'val_couleur.npy')
    if not os.path.exists(fichier):
        raise FileNotFoundError("images d'ImageNet64 absentes (%s) : dans la cellule « Données » du notebook, remplacer "
                                "if False: par if True: et l'exécuter (téléchargement de 417 Mo)" % fichier)
    tableau = np.load(fichier, mmap_mode='r')
    if indices is not None:
        x = torch.from_numpy(np.array(tableau[np.asarray(indices)])).float() / 255
    else:
        fin = None if n is None else debut + n
        x = torch.from_numpy(np.array(tableau[debut:fin])).float() / 255
    return x[:, None] if gris else x.permute(0, 3, 1, 2).contiguous()


# Les 15 images de référence des auteurs (figure 3 de l'article ; energy_histogram.ipynb) : log-probabilités
# régulièrement espacées entre la plus faible et la plus forte du jeu de test
IDX_REF_AUTEURS = [21143, 34295, 42627, 47510, 8718, 4375, 1266, 47346, 11592, 36426, 31424, 17305, 14457, 39755, 41191]


def energies_auteurs():
    '''Énergies (nats, normalisées) des 50 000 images de validation calculées par les auteurs avec le réseau en
    niveaux de gris (outputs/energies_test.pt), et leur constante de normalisation.'''
    constante, e = torch.load(os.path.join(RACINE, 'outputs', 'energies_test.pt'), map_location='cpu',
                              weights_only=False)
    return e, float(constante)


# ------------------------------------------------------------------------------------------------
# Mesures sur les images
# ------------------------------------------------------------------------------------------------
def coefficients_ondelettes(X, J=6, ondelette='db1'):
    '''Décomposition en ondelettes (Haar par défaut) sur J échelles, comme les auteurs (affine_and_sparsity.ipynb,
    multi_scale_decompose) : coefficients de détail des J échelles et passe-bas de la dernière. pywt.dwt2 est
    orthonormée ; comme les auteurs, on divise les coefficients de l'échelle j (j = 1..J) par 2**j, ce qui
    revient à des moyennes locales plutôt qu'à une base orthonormée. X : (N, 1, H, W). Renvoie (N, H*W).'''
    import pywt
    bas = X[:, 0].numpy().astype(np.float64)
    morceaux = []
    for j in range(1, J + 1):
        bas, (h, v, dd) = pywt.dwt2(bas, ondelette, mode='symmetric')
        morceaux += [c.reshape(len(X), -1) / 2 ** j for c in (h, v, dd)]
    morceaux.append(bas.reshape(len(X), -1) / 2 ** J)
    return torch.from_numpy(np.concatenate(morceaux, axis=1)).float()


def rapport_participation(X, J=6, ondelette='db1'):
    '''Rapport de participation ||c||_1^2 / (n ||c||_2^2) des coefficients d'ondelettes c (n coefficients) :
    proche de 0 pour une image parcimonieuse, 1 si tous les coefficients ont la même amplitude.'''
    c = coefficients_ondelettes(X, J, ondelette)
    return c.abs().sum(1) ** 2 / (c.shape[1] * (c ** 2).sum(1))


def ajuster_gumbel(valeurs, n_iterations=100, lr=0.01):
    '''Ajustement d'une loi de Gumbel (des maxima) par maximum de vraisemblance, comme les auteurs : initialisation
    par les moments, puis Adam. Renvoie (position, échelle).'''
    data = torch.as_tensor(valeurs, dtype=torch.float32)
    echelle = torch.sqrt(data.std() ** 2 * 6 / math.pi ** 2)
    position = data.mean() - echelle * np.euler_gamma
    position.requires_grad_(True)
    echelle.requires_grad_(True)
    opt = torch.optim.Adam([position, echelle], lr=lr)
    for _ in range(n_iterations):
        z = (data - position) / echelle
        perte = torch.mean(z + torch.exp(-z)) + torch.log(echelle)
        opt.zero_grad()
        perte.backward()
        opt.step()
    return position.item(), echelle.item()


def densite_gumbel(x, position, echelle):
    z = (np.asarray(x) - position) / echelle
    return np.exp(-z - np.exp(-z)) / echelle


def energie_effective(m, X, ts, n_bruits=8, taille_lot=64, graine=0, cache=None):
    '''Pour chaque image x de X et chaque variance t de ts : énergie effective E_y[U(y, t) | x] (nats, normalisée)
    et dimension effective d_eff(x, t) = E_y[||x - x̂(y)||^2 | x] / t (équation (11)), estimées avec n_bruits
    réalisations du bruit (graine propre à chaque niveau). Renvoie deux tableaux (len(X), len(ts)).
    Avec cache=<nom>, chaque niveau de bruit est enregistré dans resultats/cache/ (reprise après interruption).'''
    E = torch.zeros(len(X), len(ts))
    D = torch.zeros(len(X), len(ts))
    Xr = X.repeat_interleave(n_bruits, 0)
    for j, t in enumerate(ts):
        t = float(t)
        chemin = None
        if cache is not None and m.nom is not None:
            chemin = chemin_cache('deff_' + cache, modele=m.nom, donnees=empreinte(X), t=t, n_bruits=n_bruits,
                                  graine=graine)
            if os.path.exists(chemin):
                E[:, j], D[:, j] = charger(chemin)
                continue
        fixer_graine(graine * 1000 + j)
        es, ds = [], []
        for lot in Xr.split(taille_lot):
            lot = lot.to(DEVICE)
            y = lot + math.sqrt(t) * torch.randn_like(lot)
            s = m.scores(y, t)
            es.append(s.energy.detach().cpu())
            ds.append(((s.denoised.detach() - lot) ** 2).flatten(1).sum(1).cpu() / t)
        E[:, j] = torch.cat(es).view(len(X), n_bruits).mean(1)
        D[:, j] = torch.cat(ds).view(len(X), n_bruits).mean(1)
        if chemin is not None:
            sauver((E[:, j].clone(), D[:, j].clone()), chemin)
    return E, D


# ------------------------------------------------------------------------------------------------
# Mélange de deux gaussiennes (figure 1 de l'article, simple_example.ipynb des auteurs)
# ------------------------------------------------------------------------------------------------
class MelangeGaussien:
    '''Mélange sum_k p_k N(0, sigma_k^2 I_d). Sa version bruitée (variance t) est le mélange des
    N(0, (sigma_k^2 + t) I_d) : tout ne dépend que du rayon r = ||y||.'''

    def __init__(self, d=1000, sigmas=(1.0, 4.0), poids=(0.5, 0.5)):
        self.d = d
        self.sigmas = torch.tensor(sigmas, dtype=torch.float64)
        self.poids = torch.tensor(poids, dtype=torch.float64)

    def _variances(self, t):
        '''sigma_k^2 + t, de forme (..., K) ; t scalaire ou tenseur de même forme que r.'''
        t = torch.as_tensor(t, dtype=torch.float64)
        return self.sigmas ** 2 + t[..., None]

    def energies_composantes(self, r, t=0.0):
        '''-log(p_k N(y; 0, (sigma_k^2 + t) I)) pour chaque composante k : (N..., K).'''
        r = torch.as_tensor(r, dtype=torch.float64)
        v = self._variances(t)
        return r[..., None] ** 2 / (2 * v) + self.d / 2 * torch.log(2 * math.pi * v) - torch.log(self.poids)

    def energie(self, r, t=0.0):
        '''U(y, t) = -log p(y | t) (nats), fonction du rayon r = ||y||.'''
        return -torch.logsumexp(-self.energies_composantes(r, t), dim=-1)

    def responsabilites(self, r, t=0.0):
        return torch.softmax(-self.energies_composantes(r, t), dim=-1)

    def score_radial(self, r, t=0.0):
        '''Composante radiale de ∇_y U : sum_k w_k(r) r / (sigma_k^2 + t).'''
        r = torch.as_tensor(r, dtype=torch.float64)
        return (self.responsabilites(r, t) * r[..., None] / self._variances(t)).sum(-1)

    def score_temps(self, r, t=0.0):
        '''∂_t U : sum_k w_k(r) (d / 2v_k - r^2 / 2v_k^2), v_k = sigma_k^2 + t.'''
        r = torch.as_tensor(r, dtype=torch.float64)
        v = self._variances(t)
        return (self.responsabilites(r, t) * (self.d / (2 * v) - r[..., None] ** 2 / (2 * v ** 2))).sum(-1)

    def echantillons(self, n, graine=0):
        g = torch.Generator().manual_seed(graine)
        k = torch.multinomial(self.poids.float(), n, replacement=True, generator=g)
        return self.sigmas.float()[k][:, None] * torch.randn(n, self.d, generator=g)


class EBMRadial(nn.Module):
    '''Modèle d'énergie des auteurs pour le mélange (simple_example.ipynb, équation (44)) :
    U_theta(y, t) = -log sum_i exp(-a_i(t) ||y||^2 - d b_i(t)), où (a_1, a_2, b_1, b_2) sont produits par un
    perceptron à partir de log(t + t_min).'''

    def __init__(self, d, t_min, largeur=256, n_couches=4):
        super().__init__()
        self.d = d
        self.t_min = t_min
        couches = []
        for i in range(n_couches):
            couches += [nn.Linear(1 if i == 0 else largeur, largeur), nn.GELU()]
        self.model = nn.Sequential(*couches, nn.Linear(largeur, 4))

    def forward_r(self, r, t):
        c = self.model(torch.log(t + self.t_min)[:, None])
        a, b = c[:, :2], self.d * c[:, 2:]
        return -torch.logsumexp(-(a * r[:, None] ** 2 + b), dim=-1)

    def forward(self, x, t):
        return self.forward_r(torch.linalg.norm(x, dim=-1), t)


def perte_dsm(sortie, lot, d):
    '''Perte de score matching dual (équation (42)), exactement comme compute_metrics (main.py) :
    MSE de débruitage / t   +   (cible / 2 - t ∂_t U / d)^2,  avec cible = 1 - ||y - x||^2 / (d t).
    Renvoie les deux termes séparément, par exemple du lot.'''
    t = lot.noise_level.variance
    axes = tuple(range(-noise.s.signal_ndim, 0))
    espace = torch.mean((sortie.denoised - lot.clean) ** 2, dim=axes) / t
    cible = 1 - torch.mean((lot.noisy - lot.clean) ** 2, dim=axes) / t
    temps = (cible / 2 - t * sortie.noise_score / d) ** 2
    return espace, temps


def _etat_rng(g):
    return {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'generateur': g.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restaurer_rng(etat, g):
    torch.set_rng_state(etat['torch'])
    np.random.set_state(etat['numpy'])
    g.set_state(etat['generateur'])
    if etat['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(etat['cuda'])


def entrainer_ebm_radial(X, dual, n_iterations=4000, lr=1e-3, taille_lot=512, t_min=1e-2, t_max=1e2, graine=0,
                         afficher_tous=1000, cache=None, sauvegarde_tous=500):
    '''Entraîne EBMRadial sur les vecteurs X (N, d) par score matching simple (dual=False : terme d'espace seul)
    ou dual (dual=True), avec le modèle EnergyModel des auteurs, log t uniforme dans [t_min, t_max] et un pas
    divisé par deux tous les n_iterations / 10. Renvoie le modèle, la constante de normalisation (9) et
    l'historique des deux termes de la perte. Avec cache=<nom>, un point de reprise (poids, optimiseur, historique,
    états aléatoires) est enregistré tous les sauvegarde_tous itérations : une exécution interrompue reprend là où
    elle s'était arrêtée, avec le même résultat, et un entraînement terminé n'est pas refait.'''
    mode_vecteurs()
    fixer_graine(graine)
    d = X.shape[1]
    reseau = EBMRadial(d, t_min)
    modele = EnergyModel(reseau, dataset_info=None, noise_covariance=IdentityCovariance())
    opt = torch.optim.Adam(modele.parameters(), lr=lr)
    echantillonneur = UniformLog(min=NoiseLevel(variance=t_min), max=NoiseLevel(variance=t_max))
    g = torch.Generator().manual_seed(graine)
    historique, debut_i = [], 0
    chemin = None if cache is None else chemin_cache('ebm_' + cache, dual=dual, n=n_iterations, lr=lr, lot=taille_lot,
                                                     t_min=t_min, t_max=t_max, graine=graine, donnees=empreinte(X))
    if chemin is not None and os.path.exists(chemin):
        etat = charger(chemin)
        reseau.load_state_dict(etat['reseau'])
        if etat['fini']:
            print('  entraînement déjà fait (cache %s)' % os.path.basename(chemin))
            return reseau, etat['constante'], np.asarray(etat['historique'])
        opt.load_state_dict(etat['opt'])
        historique, debut_i = etat['historique'], etat['i']
        _restaurer_rng(etat['rng'], g)
        print('  reprise à l\'itération %d' % debut_i)
    debut = time.time()
    for i in range(debut_i, n_iterations):
        if i % (n_iterations // 10) == 0:
            for groupe in opt.param_groups:
                groupe['lr'] = lr / 2 ** (i // (n_iterations // 10))
        lot = noisy_batch(X[torch.randint(0, len(X), (taille_lot,), generator=g)], echantillonneur)
        sortie = modele(ModelInput.from_batch(lot))
        espace, temps = perte_dsm(sortie, lot, d)
        perte = (espace + temps).mean() if dual else espace.mean()
        opt.zero_grad()
        perte.backward()
        opt.step()
        historique.append((espace.mean().item(), temps.mean().item()))
        if afficher_tous and (i + 1) % afficher_tous == 0:
            print('  itération %5d  terme d\'espace %.4f  terme de temps %.4f  (%.0f s)' % (
                i + 1, historique[-1][0], historique[-1][1], time.time() - debut), flush=True)
        if chemin is not None and (i + 1) % sauvegarde_tous == 0 and i + 1 < n_iterations:
            sauver({'reseau': reseau.state_dict(), 'opt': opt.state_dict(), 'historique': historique, 'i': i + 1,
                    'rng': _etat_rng(g), 'fini': False}, chemin)
    with torch.no_grad():
        y = X[:20000] + math.sqrt(t_max) * torch.randn_like(X[:20000])
        tt = torch.full((len(y),), t_max)
        constante = d / 2 * math.log(2 * math.pi * math.e * t_max) - reseau(y, tt).mean().item()
    if chemin is not None:
        sauver({'reseau': reseau.state_dict(), 'historique': historique, 'constante': constante, 'fini': True}, chemin)
    return reseau, constante, np.asarray(historique)


# ------------------------------------------------------------------------------------------------
# Entraînement de petits réseaux d'images (notebook TP_entrainement_dual)
# ------------------------------------------------------------------------------------------------
def sous_echantillonner(X, facteur):
    '''Réduction de résolution par moyenne sur des blocs facteur x facteur (filtre « boîte », comme
    Chrabaszcz et al. pour ImageNet64).'''
    return torch.nn.functional.avg_pool2d(X, facteur) if facteur > 1 else X


def construire_reseau(info, type_reseau, base_width=16, num_scales=2, n_couches=2, graine=0):
    '''UNet des auteurs (sans biais, normalisation d'instance homogène, GELU, conditionnement par t) et sa
    reparamétrisation : « score » -> débruiteur y - t s(y, t) (réseau de score des auteurs) ;
    « energie » -> U = <y, s(y, t)> / 2. L'initialisation dépend de la graine.'''
    fixer_graine(graine)
    unet = UNet(dataset_info=info, num_scales=num_scales, base_width=base_width, group_size=1,
                num_layers_encoder_block=n_couches, num_layers_mid_block=n_couches,
                num_layers_decoder_block=n_couches)
    if type_reseau == 'score':
        enveloppe = Reparameterization(unet, dataset_info=info, output_scaling=1, residual=True)
    else:
        enveloppe = Reparameterization(unet, dataset_info=info, conversion='inner_product')
    m = Modele.depuis_reseau(enveloppe, info, est_energie=type_reseau != 'score')
    m.description = dict(type=type_reseau, base_width=base_width, num_scales=num_scales, n_couches=n_couches,
                         graine=graine, K=info.spatial_size)
    return m


def entrainer_reseau(m, X, perte, n_iterations=1500, taille_lot=64, lr=5e-4, t_min=T_MIN, t_max=T_MAX, graine=0,
                     afficher_tous=250, cache=None, sauvegarde_tous=100, n_divisions=10):
    '''Entraîne le réseau m (construire_reseau) sur les images X (N, C, H, W), comme main.py : log t uniforme
    dans [t_min, t_max], Adam, pas divisé par deux n_divisions fois à intervalles réguliers (10 chez les auteurs).
    perte : « dsm » (score matching ordinaire : terme d'espace seul) ou « dual » (espace + temps). Renvoie l'historique (terme d'espace,
    terme de temps, durée de l'itération). Avec cache=<nom>, un point de reprise est enregistré tous les
    sauvegarde_tous itérations (reprise après interruption, même résultat) ; un entraînement terminé n'est pas
    refait (les durées enregistrées sont celles de l'entraînement initial).'''
    mode_images()
    fixer_graine(graine)
    opt = torch.optim.Adam(m.reseau.parameters(), lr=lr)
    echantillonneur = UniformLog(min=NoiseLevel(m.info, variance=t_min), max=NoiseLevel(m.info, variance=t_max))
    g = torch.Generator().manual_seed(graine)
    historique, debut_i = [], 0
    chemin = None if cache is None else chemin_cache(
        'reseau_' + cache, description=getattr(m, 'description', None), perte=perte, n=n_iterations, lot=taille_lot,
        lr=lr, t_min=t_min, t_max=t_max, graine=graine, donnees=empreinte(X), n_divisions=n_divisions)
    enveloppe = m.reseau.network
    palier = max(1, n_iterations // n_divisions)
    if chemin is not None and os.path.exists(chemin):
        etat = charger(chemin)
        enveloppe.load_state_dict(etat['reseau'])
        historique = etat['historique']
        if etat['fini']:
            print('  entraînement déjà fait (cache %s)' % os.path.basename(chemin))
            debut_i = n_iterations
        else:
            opt.load_state_dict(etat['opt'])
            debut_i = etat['i']
            _restaurer_rng(etat['rng'], g)
            print('  reprise à l\'itération %d' % debut_i)
    m.reseau.train()
    debut = time.time()
    for i in range(debut_i, n_iterations):
        for groupe in opt.param_groups:
            groupe['lr'] = lr / 2 ** (i // palier)
        t0 = time.time()
        lot = noisy_batch(X[torch.randint(0, len(X), (taille_lot,), generator=g)].to(DEVICE), echantillonneur)
        sortie = m.reseau(ModelInput.from_batch(lot), create_graph=True) if m.est_energie else \
            m.reseau(ModelInput.from_batch(lot))
        t = lot.noise_level.variance
        espace = torch.mean((sortie.denoised - lot.clean) ** 2, dim=(-1, -2, -3)) / t
        if m.est_energie:
            cible = 1 - torch.mean((lot.noisy - lot.clean) ** 2, dim=(-1, -2, -3)) / t
            temps = (cible / 2 - t * sortie.noise_score / m.d) ** 2
        else:
            temps = torch.zeros_like(espace)
        p = (espace + temps).mean() if perte == 'dual' else espace.mean()
        opt.zero_grad()
        p.backward()
        opt.step()
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        historique.append((espace.mean().item(), temps.mean().item(), time.time() - t0))
        if afficher_tous and (i + 1) % afficher_tous == 0:
            h = np.asarray(historique[-afficher_tous:])
            print('  itération %5d  terme d\'espace %.4f  terme de temps %.4f  (%.0f s)' % (
                i + 1, h[:, 0].mean(), h[:, 1].mean(), time.time() - debut), flush=True)
        if chemin is not None and (i + 1) % sauvegarde_tous == 0 and i + 1 < n_iterations:
            sauver({'reseau': enveloppe.state_dict(), 'opt': opt.state_dict(), 'historique': historique, 'i': i + 1,
                    'rng': _etat_rng(g), 'fini': False}, chemin)
    if chemin is not None and debut_i < n_iterations:
        sauver({'reseau': enveloppe.state_dict(), 'historique': historique, 'fini': True}, chemin)
    m.reseau.eval()
    for q in m.reseau.parameters():
        q.requires_grad = False
    return np.asarray(historique)


# ------------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------------
_FIGURES = {'prefixe': None, 'dossier': None}


def nommer_figures(nom_notebook, dossier=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultats',
                                                       'figures')):
    '''Les figures montrées par montrer(nom) sont enregistrées sous <dossier>/<nom_notebook>_fig_<nom>.png.'''
    os.makedirs(dossier, exist_ok=True)
    _FIGURES['prefixe'] = nom_notebook
    _FIGURES['dossier'] = dossier


def montrer(nom=None, fig=None):
    fig = fig or plt.gcf()
    fig.tight_layout()
    if nom and _FIGURES['prefixe']:
        fig.savefig(os.path.join(_FIGURES['dossier'], '%s_fig_%s.png' % (_FIGURES['prefixe'], nom)), dpi=130,
                    bbox_inches='tight')
    plt.show()


def afficher_images(images, titres=None, ncol=10, taille=1.4, titre=None, nom=None, vmin=0.0, vmax=1.0):
    '''Grille d'images (N, C, H, W) dans [0, 1].'''
    images = images.detach().cpu()
    n = len(images)
    nlig = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nlig, ncol, figsize=(taille * ncol, taille * nlig + (0.3 if titre else 0)),
                             squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i < n:
            im = images[i]
            if im.shape[0] == 1:
                ax.imshow(im[0], cmap='gray', vmin=vmin, vmax=vmax)
            else:
                ax.imshow(im.permute(1, 2, 0).clamp(0, 1))
            if titres is not None:
                ax.set_title(titres[i], fontsize=8)
    if titre:
        fig.suptitle(titre)
    montrer(nom, fig)
