#!/usr/bin/env python3
"""
Hessian V1:
nauceni skor cele kombinacije i koordinisane promene dva broja.

Zavisnosti: numpy, scipy
CSV: sedam sortiranih brojeva, bez zaglavlja.
Redosled: najstarije kolo prvo, najnovije poslednje.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


DEFAULTS = (
    "/data/loto7_4700_k81.csv",
    "/data/loto7_4700_k81_loto_2972.csv",
    "/data/loto7_4700_k81_loto_plus_1728.csv",
)

WINDOW = 96
STARTS = 12
PAIR_KEEP = 32

ALPHAS = (.01, .1, 1.)
DIM = 98

I, J = np.triu_indices(7, 1)

# Koraci odredjuju opseg pretrage, ne raspodelu brojeva.
PAIR_STEPS = np.array((-4, -2, -1, 1, 2, 4))

A, B = np.meshgrid(
    PAIR_STEPS,
    PAIR_STEPS,
    indexing="ij"
)

MOVE_A = A.ravel()
MOVE_B = B.ravel()


def load(path):
    rows = []

    with Path(path).open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not v.strip() for v in row):
                continue

            try:
                values = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem cele brojeve."
                ) from error

            if (
                len(values) != 7
                or values != sorted(values)
                or len(set(values)) != 7
                or not all(1 <= v <= 39 for v in values)
            ):
                raise ValueError(
                    f"Red {line}: neispravna kombinacija 7/39."
                )

            rows.append(values)

    return np.asarray(rows, dtype=int).reshape(-1, 7)


def gaps(rows):
    rows = np.atleast_2d(rows)

    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), 40)
    ))

    return np.diff(edges, axis=1) - 1.


def replacements(row):
    unused = np.setdiff1d(np.arange(1, 40), row)
    result = np.tile(row, (224, 1))

    for j in range(7):
        result[j * 32:(j + 1) * 32, j] = unused

    return np.sort(result, axis=1)


def support(history):
    tree = cKDTree(gaps(history) / 5.)

    distance = tree.query(
        gaps(history) / 5.,
        k=2,
        p=1
    )[0][:, 1]

    return tree, max(
        float(np.quantile(distance, .75)),
        1e-12
    )


def allowed(rows, tree, radius):
    distance = tree.query(
        gaps(rows) / 5., p=1
    )[0]

    return distance <= radius + 1e-10


def features(g, reference, memory):
    # Osmi razmak odredjen je fiksnim zbirom 32.
    q = (
        np.atleast_2d(g)[:, :7] - reference[:7]
    ) / 5.

    m = memory[:7] / 5.

    return np.column_stack((
        q,
        q * q,
        q[:, I] * q[:, J],
        q ** 3,
        q ** 4,
        (
            q[:, :, None] * m[None, None, :]
        ).reshape(-1, 49)
    ))


def prepare(history):
    if len(history) < WINDOW + 2:
        raise ValueError(
            f"Potrebno je najmanje "
            f"{WINDOW + 2} prethodnih redova."
        )

    h = history[-(WINDOW + 2):]
    g = gaps(h)

    gram = np.zeros((DIM, DIM))
    rhs = np.zeros(DIM)

    for t in range(2, len(h)):
        memory = g[t - 1] - g[t - 2]

        actual = features(
            g[t:t + 1], g[t - 1], memory
        )

        other = features(
            gaps(replacements(h[t])),
            g[t - 1],
            memory
        )

        contrast = actual - other

        gram += (
            contrast.T @ contrast / len(contrast)
        )

        rhs += contrast.mean(axis=0)

    gram /= WINDOW
    rhs /= WINDOW

    # Numericka skala osobina.
    # Nema prosecne ciljne kombinacije.
    scale = np.maximum(
        np.sqrt(np.diag(gram)), 1e-8
    )

    return (
        gram / scale[:, None] / scale[None, :],
        rhs / scale,
        scale,
        g[-1],
        g[-1] - g[-2],
        h[-WINDOW:]
    )


def fit(p, alpha):
    coefficient = (
        np.linalg.solve(
            p[0] + alpha * np.eye(DIM),
            p[1]
        ) / p[2]
    )

    linear = (
        coefficient[:7]
        + coefficient[49:].reshape(7, 7)
        @ (p[4][:7] / 5.)
    )

    h0 = np.diag(
        2. * coefficient[7:14]
    )

    h0[I, J] = coefficient[14:35]
    h0[J, I] = coefficient[14:35]

    return (
        coefficient,
        p[3],
        p[4],
        linear,
        h0,
        coefficient[35:42],
        coefficient[42:49]
    )


def coordinates(rows, model):
    return (
        gaps(rows)[:, :7] - model[1][:7]
    ) / 5.


def polynomial(q, model):
    q = np.atleast_2d(q)

    (
        _, _, _,
        linear,
        h0,
        cubic,
        quartic
    ) = model

    return (
        q @ linear
        + .5 * np.einsum(
            "ni,ij,nj->n", q, h0, q
        )
        + (q ** 3) @ cubic
        + (q ** 4) @ quartic
    )


def scores(rows, model):
    value = polynomial(
        coordinates(rows, model),
        model
    )

    if not np.all(np.isfinite(value)):
        raise ArithmeticError("Neispravan skor.")

    return value


def derivatives(q, model):
    (
        _, _, _,
        linear,
        h0,
        cubic,
        quartic
    ) = model

    gradient = (
        linear
        + h0 @ q
        + 3. * cubic * q * q
        + 4. * quartic * q ** 3
    )

    hessian = (
        h0
        + np.diag(
            6. * cubic * q
            + 12. * quartic * q * q
        )
    )

    return gradient, hessian


def pair_candidates(row):
    block = len(MOVE_A)

    result = np.tile(
        row, (len(I) * block, 1)
    )

    for k, (i, j) in enumerate(zip(I, J)):
        selected = slice(
            k * block, (k + 1) * block
        )

        result[selected, i] += MOVE_A
        result[selected, j] += MOVE_B

    result = np.sort(result, axis=1)

    valid = (
        (result[:, 0] >= 1)
        & (result[:, -1] <= 39)
        & np.all(
            np.diff(result, axis=1) > 0,
            axis=1
        )
        & np.any(result != row, axis=1)
    )

    return np.unique(
        result[valid], axis=0
    )


def guided_pairs(row, model, tree, radius):
    candidates = pair_candidates(row)

    candidates = candidates[
        allowed(candidates, tree, radius)
    ]

    if not len(candidates):
        return candidates

    q = coordinates(row, model)[0]

    delta = (
        coordinates(candidates, model) - q
    )

    gradient, hessian = derivatives(q, model)

    predicted_gain = (
        delta @ gradient
        + .5 * np.einsum(
            "ni,ij,nj->n",
            delta,
            hessian,
            delta
        )
    )

    order = np.argsort(
        -predicted_gain,
        kind="stable"
    )[:PAIR_KEEP]

    return candidates[order]


def climb_one(row, model, tree, radius):
    row = row.copy()
    steps = 0

    while True:
        candidates = replacements(row)

        candidates = candidates[
            allowed(candidates, tree, radius)
        ]

        if not len(candidates):
            break

        values = scores(candidates, model)
        best = int(np.argmax(values))

        if (
            values[best]
            <= scores(row, model)[0] + 1e-10
        ):
            break

        row = candidates[best].copy()
        steps += 1

    return row, steps


def search(history, model):
    tree, radius = support(history)

    seeds = np.unique(history, axis=0)

    order = np.argsort(
        -scores(seeds, model),
        kind="stable"
    )[:STARTS]

    current = seeds[order].copy()
    steps = 0

    # Kontrolna pretraga: sve zamene jednog broja.
    for j, row in enumerate(current):
        current[j], count = climb_one(
            row, model, tree, radius
        )

        steps += count

    values = scores(current, model)

    baseline = tuple(map(
        int,
        current[int(np.argmax(values))]
    ))

    # Hessian nastavlja od istih zavrsnih tacaka.
    while True:
        moved = False

        for j, row in enumerate(current):
            candidates = guided_pairs(
                row, model, tree, radius
            )

            if not len(candidates):
                continue

            values = scores(candidates, model)
            best = int(np.argmax(values))

            # Prihvatanje koristi TACAN skor,
            # a ne Hessian aproksimaciju.
            if (
                values[best]
                > scores(row, model)[0] + 1e-10
            ):
                current[j], count = climb_one(
                    candidates[best],
                    model,
                    tree,
                    radius
                )

                steps += count + 1
                moved = True

        if not moved:
            break

    values = scores(current, model)
    best = int(np.argmax(values))

    prediction = tuple(map(
        int, current[best]
    ))

    assert (
        values[best]
        >= scores(
            np.array(baseline), model
        )[0] - 1e-9
    )

    return (
        prediction,
        float(values[best]),
        steps,
        baseline
    )


def predict(p, alpha):
    return search(
        p[5], fit(p, alpha)
    )


def hits(prediction, actual):
    return len(
        set(prediction).intersection(actual)
    )


def run_file(path, validation=48, test=64):
    history = load(path)
    split = len(history) - test

    if (
        validation < 16
        or test < 16
        or split - validation < WINDOW + 2
    ):
        raise ValueError(
            "Nedovoljno podataka ili prekratka provera."
        )

    totals = np.zeros(
        len(ALPHAS), dtype=int
    )

    # Parametar se bira pre odvojenog testa.
    for t in range(
        split - validation, split
    ):
        p = prepare(history[:t])

        for j, alpha in enumerate(ALPHAS):
            totals[j] += hits(
                predict(p, alpha)[0],
                history[t]
            )

    alpha = ALPHAS[
        int(np.argmax(totals))
    ]

    joint = []
    single = []
    previous = []
    changed = 0

    # Obe pretrage koriste isti nauceni model.
    # Parametar je zamrznut tokom testa.
    for t in range(split, len(history)):
        prediction, _, _, baseline = predict(
            prepare(history[:t]),
            alpha
        )

        joint.append(
            hits(prediction, history[t])
        )

        single.append(
            hits(baseline, history[t])
        )

        previous.append(
            hits(history[t - 1], history[t])
        )

        changed += prediction != baseline

    print("\nCSV:", path)

    print(
        "Hronologija: CSV mora biti od najstarijeg "
        "ka najnovijem kolu."
    )

    print(
        f"Validacija: {validation}; "
        f"odvojeni test: {test}; "
        f"alpha={alpha:g}"
    )

    for label, values in (
        (
            "HESSIAN: KOORDINISANE PROMENE",
            joint
        ),
        (
            "Kontrola: isti skor, samo zamene jednog broja",
            single
        ),
        (
            "Kontrola: prethodno izvuceno kolo",
            previous
        )
    ):
        print(
            f"{label}: "
            f"prosek={np.mean(values):.4f}; "
            f"pogoci 0..7="
            f"{np.bincount(values, minlength=8).tolist()}"
        )

    print(
        f"Hessian promenio izbor u "
        f"{changed}/{test} testnih kola."
    )

    prediction, score, steps, baseline = predict(
        prepare(history), alpha
    )

    replay = bool(
        np.any(
            np.all(
                history == prediction,
                axis=1
            )
        )
    )

    print(
        f"SLEDECI RED {len(history) + 1}:",
        ",".join(map(str, prediction))
    )

    print(
        f"Skor={score:.10f}; "
        f"svi koraci={steps}; "
        f"istorijsko ponavljanje={replay}"
    )

    print(
        "Kontrolni izbor:",
        ",".join(map(str, baseline))
    )

    print(
        f"Opseg: {STARTS} istorijskih polazista, "
        "sve zamene jednog broja,"
    )

    print(
        f"parovi sa koracima {PAIR_STEPS.tolist()}, "
        f"do {PAIR_KEEP} predloga po tacki."
    )

    print(
        "Skor nije verovatnoca; "
        "globalni maksimum nije potvrden."
    )

    return prediction


def self_test():
    base = np.array((
        (1, 2, 3, 4, 5, 6, 7),
        (4, 12, 18, 28, 35, 36, 37),
        (33, 34, 35, 36, 37, 38, 39),
        (4, 8, 13, 18, 23, 28, 34)
    ))

    g = gaps(base)

    assert np.all(
        g.sum(axis=1) == 32
    )

    assert np.array_equal(
        np.cumsum(
            g[:, :-1] + 1,
            axis=1
        ),
        base
    )

    h = np.tile(base, (32, 1))
    p = prepare(h)
    model = fit(p, .1)

    scaled = model[0] * p[2]

    assert np.allclose(
        (
            p[0] + .1 * np.eye(DIM)
        ) @ scaled,
        p[1]
    )

    assert np.allclose(
        scores(base, model),
        features(
            g, p[3], p[4]
        ) @ model[0]
    )

    q = np.array((
        .1, -.2, .3, -.1, .2, .1, -.3
    ))

    gradient, hessian = derivatives(q, model)

    eps = 1e-5
    numerical_gradient = np.zeros(7)
    numerical_hessian = np.zeros((7, 7))

    for j in range(7):
        step = np.eye(7)[j] * eps

        numerical_gradient[j] = (
            polynomial(q + step, model)[0]
            - polynomial(q - step, model)[0]
        ) / (2 * eps)

        numerical_hessian[:, j] = (
            derivatives(q + step, model)[0]
            - derivatives(q - step, model)[0]
        ) / (2 * eps)

    assert np.allclose(
        gradient,
        numerical_gradient,
        rtol=1e-6,
        atol=1e-7
    )

    assert np.allclose(
        hessian,
        numerical_hessian,
        rtol=1e-6,
        atol=1e-7
    )

    assert np.array_equal(
        hessian, hessian.T
    )

    prediction, score, _, baseline = predict(
        p, .1
    )

    assert prediction == predict(
        prepare(h.copy()), .1
    )[0]

    assert (
        len(set(prediction)) == 7
        and 1 <= min(prediction)
        <= max(prediction) <= 39
    )

    assert (
        score >= scores(
            np.array(baseline), model
        )[0] - 1e-9
    )

    tree, radius = support(p[5])

    other = replacements(
        np.array(prediction)
    )

    other = other[
        allowed(other, tree, radius)
    ]

    assert (
        scores(
            other, model
        ).max(initial=-np.inf)
        <= score + 1e-9
    )

    pairs = guided_pairs(
        np.array(prediction),
        model,
        tree,
        radius
    )

    assert (
        scores(
            pairs, model
        ).max(initial=-np.inf)
        <= score + 1e-9
    )

    future = h.copy()
    future[-1] = base[1]

    assert predict(
        prepare(h[:-1]), .1
    )[0] == predict(
        prepare(future[:-1]), .1
    )[0]

    print(
        "Self-test OK: gradijent, Hessian, "
        "tacan skor, pretraga i hronologija."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "csv",
        nargs="*",
        help="Podrazumevano sva tri CSV fajla."
    )

    parser.add_argument(
        "--validation",
        type=int,
        default=48
    )

    parser.add_argument(
        "--test",
        type=int,
        default=64
    )

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()

    for path in args.csv or DEFAULTS:
        run_file(
            path,
            args.validation,
            args.test
        )


if __name__ == "__main__":
    main()



"""
CSV: /data/loto7_4700_k81.csv
Hronologija: CSV mora biti od najstarijeg ka najnovijem kolu.
Validacija: 48; odvojeni test: 64; alpha=0.01
HESSIAN: KOORDINISANE PROMENE: prosek=1.1719; pogoci 0..7=[9, 40, 11, 3, 1, 0, 0, 0]
Kontrola: isti skor, samo zamene jednog broja: prosek=1.1719; pogoci 0..7=[10, 38, 12, 3, 1, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.2344; pogoci 0..7=[13, 30, 15, 5, 1, 0, 0, 0]
Hessian promenio izbor u 12/64 testnih kola.
SLEDECI RED 4701: 3,x,8,y,10,z,39
Skor=6.1039209085; svi koraci=103; istorijsko ponavljanje=False
Kontrolni izbor: 5,9,10,11,12,13,39
Opseg: 12 istorijskih polazista, sve zamene jednog broja,
parovi sa koracima [-4, -2, -1, 1, 2, 4], do 32 predloga po tacki.
Skor nije verovatnoca; globalni maksimum nije potvrden.

CSV: /data/loto7_4700_k81_loto_2972.csv
Hronologija: CSV mora biti od najstarijeg ka najnovijem kolu.
Validacija: 48; odvojeni test: 64; alpha=0.01
HESSIAN: KOORDINISANE PROMENE: prosek=1.2969; pogoci 0..7=[17, 21, 17, 8, 1, 0, 0, 0]
Kontrola: isti skor, samo zamene jednog broja: prosek=1.3281; pogoci 0..7=[16, 21, 18, 8, 1, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.3281; pogoci 0..7=[10, 30, 17, 7, 0, 0, 0, 0]
Hessian promenio izbor u 14/64 testnih kola.
SLEDECI RED 2973: 32,x,35,y,37,z,39
Skor=2.7680807452; svi koraci=123; istorijsko ponavljanje=False
Kontrolni izbor: 32,34,35,36,37,38,39
Opseg: 12 istorijskih polazista, sve zamene jednog broja,
parovi sa koracima [-4, -2, -1, 1, 2, 4], do 32 predloga po tacki.
Skor nije verovatnoca; globalni maksimum nije potvrden.

CSV: /data/loto7_4700_k81_loto_plus_1728.csv
Hronologija: CSV mora biti od najstarijeg ka najnovijem kolu.
Validacija: 48; odvojeni test: 64; alpha=0.01
HESSIAN: KOORDINISANE PROMENE: prosek=1.3125; pogoci 0..7=[10, 30, 18, 6, 0, 0, 0, 0]
Kontrola: isti skor, samo zamene jednog broja: prosek=1.3281; pogoci 0..7=[10, 30, 17, 7, 0, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.2969; pogoci 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Hessian promenio izbor u 10/64 testnih kola.
SLEDECI RED 1729: 1,x,3,y,17,z,22
Skor=1.7094543638; svi koraci=138; istorijsko ponavljanje=False
Kontrolni izbor: 1,2,3,4,17,18,22
Opseg: 12 istorijskih polazista, sve zamene jednog broja,
parovi sa koracima [-4, -2, -1, 1, 2, 4], do 32 predloga po tacki.
Skor nije verovatnoca; globalni maksimum nije potvrden.
"""



"""
Hessian matrica korisna za zajedničke promene više delova kombinacije.
Dijagonalni elementi opisuju zakrivljenost u pojedinačnim pravcima, a ostali međusobni uticaj promena. 
Koristi se u lokalnoj kvadratnoj aproksimaciji i optimizaciji. MIT — Hessian i optimizacija

Za moj Loto model izdvojio bih pretragu koordinisanih promena. 
Promena jednog broja može smanjiti skor, dok istovremena promena dva broja može povećati skor zbog njihove interakcije.

1. Učim gladak skor cele kombinacije, uslovljen prethodnim prelazima rasporeda razmaka.
2. Računam gradijent i Hessian u sedam nezavisnih dimenzija razmaka.
3. Njima usmeravam predloge zajedničkih promena, zatim ocenjujemo konkretne validne sedmorke.
4. Poredim rezultat sa pretragom koja menja samo jedan broj, uz potpuno isti naučeni skor.

Hessian je već bio u kvadratnom potencijalu Tomonaga modela. 
Ovde bi nova uloga bila vođenje pretrage kroz zajedničke promene, pa poboljšanje moram pripisati i proveriti upravo toj razlici. 
Kvadratna aproksimacija skora ne zahteva normalnu raspodelu.


Uči gladak skor cele kombinacije, a gradijent i Hessian će usmeravati zajedničke promene dva broja. 


Proverena na 48 validacionih i 64 odvojena testna kola. Hessian pretraga povećava ili zadržava naučeni skor, ali na ovom testu nije poboljšala prosečan broj pogodaka u odnosu na pretragu jednog broja.

CSV	        Hessian prosek	Jedan broj	Sledeća kombinacija
Zbirni 4700	1,1719	        1,1719	    3,x,8,y,10,z,39
Loto 2972	1,2969	        1,3281	    32,x,35,y,37,z,39
Plus 1728	1,3125	        1,3281	    1,x,3,y,17,z,22
"""
