#!/usr/bin/env python3
"""
Hessian V2:
empirijski gladak skor celih prelaza,
bez polinomske ekstrapolacije.

Zavisnosti: numpy, scipy.
CSV: sedam sortiranih brojeva, bez zaglavlja.
Najstarije kolo prvo, najnovije poslednje.
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

WINDOW = 192
STARTS = 8
PAIR_KEEP = 24

# Sirina empirijske slicnosti i doprinos celog rasporeda.
PARAMS = (
    (.5, .05),
    (.75, .1),
    (1., .1),
)

I, J = np.triu_indices(7, 1)

PAIR_STEPS = np.array((-4, -2, -1, 1, 2, 4))

A, B = np.meshgrid(
    PAIR_STEPS,
    PAIR_STEPS,
    indexing="ij",
)

MOVE_A = A.ravel()
MOVE_B = B.ravel()

FD_STEP = 1e-3


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
        np.full(len(rows), 40),
    ))

    return np.diff(edges, axis=1) - 1.


def replacements(row):
    unused = np.setdiff1d(np.arange(1, 40), row)
    result = np.tile(row, (224, 1))

    for j in range(7):
        result[j * 32:(j + 1) * 32, j] = unused

    return np.sort(result, axis=1)


def hits(prediction, actual):
    return len(set(prediction).intersection(actual))


def geometry(g):
    g = np.atleast_2d(g)

    current = np.ones(len(g), dtype=int)
    longest = current.copy()

    for j in range(1, 7):
        current = np.where(
            g[:, j] == 0,
            current + 1,
            1,
        )
        longest = np.maximum(longest, current)

    return np.column_stack((
        longest,
        np.max(g[:, 1:7], axis=1),
    ))


def field(g):
    g = np.atleast_2d(g)
    interior = g[:, 1:7]

    # Gladak opis zbijanja; nema zabrane uzastopnih brojeva.
    close = 1. / (1. + interior) ** 2

    runs = []

    for length in range(1, 7):
        value = np.ones((
            len(g), 7 - length
        ))

        for j in range(length):
            value *= close[
                :, j:j + 7 - length
            ]

        runs.append(
            value.sum(axis=1)
        )

    # Ceo raspored, zbijanje i unutrasnje praznine.
    return np.column_stack((
        g,
        np.column_stack(runs),
        np.sum(interior * interior, axis=1),
    ))


def squared(a, b):
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)

    return np.maximum(
        (a * a).sum(axis=1)[:, None]
        + (b * b).sum(axis=1)[None, :]
        - 2. * a @ b.T,
        0.,
    )


def bandwidth(values):
    nearest = cKDTree(values).query(
        values, k=2
    )[0][:, 1]

    positive = nearest[
        nearest > 1e-10
    ]

    value = (
        float(np.median(positive))
        if len(positive)
        else 1.
    )

    return max(value, 1.)


def prepare(history):
    if len(history) < WINDOW + 3:
        raise ValueError(
            f"Potrebno je najmanje "
            f"{WINDOW + 3} prethodnih redova."
        )

    h = history[-(WINDOW + 3):]
    g = gaps(h)
    raw = field(g)

    floor = np.array(
        [1.] * 8 + [.02] * 6 + [1.]
    )

    # Skale osobina, bez prosecne ciljne kombinacije.
    unit = np.maximum(
        np.median(
            np.abs(np.diff(raw, axis=0)),
            axis=0,
        ),
        floor,
    )

    f = raw / unit

    target = f[3:]
    reference = f[2:-1]

    previous = f[2:-1] - f[1:-2]
    earlier = f[1:-2] - f[:-3]

    context = np.column_stack((
        reference,
        previous,
        .5 * earlier,
    ))

    current = np.concatenate((
        f[-1],
        f[-1] - f[-2],
        .5 * (f[-2] - f[-3]),
    ))

    delta = target - reference

    # Tacni ostvareni prelazi, preneti na trenutno stanje.
    translated = (
        g[-1] + (g[3:] - g[2:-1])
    )

    valid = np.all(
        translated >= 0, axis=1
    )

    transferred = np.cumsum(
        translated[valid, :7] + 1,
        axis=1,
    ).astype(int)

    seeds = np.unique(
        np.vstack((
            h[-WINDOW:],
            transferred,
        )),
        axis=0,
    )

    return dict(
        unit=unit,
        reference=f[-1],
        targets=target,
        delta=delta,
        context_distance=squared(
            current, context
        )[0],
        context_width=bandwidth(context),
        transition_width=bandwidth(delta),
        state_width=bandwidth(target),
        seeds=seeds,
    )


def fit(p, param):
    width, state_weight = param

    weights = (
        1. + p["context_distance"]
        / (p["context_width"] * width) ** 2
    ) ** -4

    weights /= weights.sum()

    return p, width, state_weight, weights


def score_g(g, model):
    p, width, state_weight, weights = model

    f = field(g) / p["unit"]

    transition_distance = squared(
        f - p["reference"],
        p["delta"],
    )

    transition_distance /= (
        p["transition_width"] * width
    ) ** 2

    conditional = (
        (1. + transition_distance) ** -4
    ) @ weights

    state_distance = squared(
        f, p["targets"]
    ) / (p["state_width"] * width) ** 2

    state = np.mean(
        (1. + state_distance) ** -4,
        axis=1,
    )

    tiny = np.finfo(float).tiny

    # Ogranicena empirijska podrska celim prelazima.
    # Skor ne raste prema ekstremima kao polinom.
    return (
        np.log(np.maximum(conditional, tiny))
        + state_weight
        * np.log(np.maximum(state, tiny))
    )


def scores(rows, model):
    rows = np.atleast_2d(rows)

    values = score_g(
        gaps(rows), model
    )

    if not np.all(np.isfinite(values)):
        raise ArithmeticError(
            "Neispravan empirijski skor."
        )

    return values


def score_q(q, model):
    q = np.atleast_2d(q)

    # Sedam nezavisnih razmaka; osmi cuva zbir 32.
    g = np.column_stack((
        q,
        32. - q.sum(axis=1),
    ))

    return score_g(g, model)


def numerical_derivatives(fun, q, step=FD_STEP):
    points = [q]

    for i in range(7):
        d = np.eye(7)[i] * step
        points.extend((q + d, q - d))

    for i, j in zip(I, J):
        a = np.eye(7)[i] * step
        b = np.eye(7)[j] * step

        points.extend((
            q + a + b,
            q + a - b,
            q - a + b,
            q - a - b,
        ))

    values = fun(np.asarray(points))

    gradient = np.zeros(7)
    hessian = np.zeros((7, 7))

    for i in range(7):
        plus, minus = values[
            1 + 2 * i:3 + 2 * i
        ]

        gradient[i] = (
            plus - minus
        ) / (2. * step)

        hessian[i, i] = (
            plus - 2. * values[0] + minus
        ) / (step * step)

    offset = 15

    for i, j in zip(I, J):
        pp, pm, mp, mm = values[
            offset:offset + 4
        ]

        value = (
            pp - pm - mp + mm
        ) / (4. * step * step)

        hessian[i, j] = value
        hessian[j, i] = value

        offset += 4

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
            axis=1,
        )
        & np.any(result != row, axis=1)
    )

    return np.unique(
        result[valid], axis=0
    )


def guided_pairs(row, model):
    candidates = pair_candidates(row)

    if not len(candidates):
        return candidates

    q = gaps(row)[0, :7]

    gradient, hessian = numerical_derivatives(
        lambda x: score_q(x, model),
        q,
    )

    delta = gaps(candidates)[:, :7] - q

    gain = (
        delta @ gradient
        + .5 * np.einsum(
            "ni,ij,nj->n",
            delta,
            hessian,
            delta,
        )
    )

    order = np.argsort(
        -gain, kind="stable"
    )[:PAIR_KEEP]

    return candidates[order]


def search(model):
    p = model[0]
    cache = {}

    def evaluate(rows):
        rows = np.atleast_2d(rows)

        keys = [
            tuple(map(int, row))
            for row in rows
        ]

        missing = sorted(set(
            key for key in keys
            if key not in cache
        ))

        if missing:
            values = scores(
                np.asarray(missing),
                model,
            )

            cache.update(zip(
                missing,
                map(float, values),
            ))

        return np.asarray([
            cache[key] for key in keys
        ])

    def climb(row):
        row = row.copy()
        steps = 0

        while True:
            candidates = replacements(row)
            values = evaluate(candidates)
            best = int(np.argmax(values))

            if (
                values[best]
                <= evaluate(row)[0] + 1e-10
            ):
                return row, steps

            row = candidates[best].copy()
            steps += 1

    seeds = p["seeds"]

    order = np.argsort(
        -evaluate(seeds),
        kind="stable",
    )[:STARTS]

    current = seeds[order].copy()
    steps = 0

    # Kontrola: potpuna pretraga zamena jednog broja.
    for j, row in enumerate(current):
        current[j], count = climb(row)
        steps += count

    current = np.unique(current, axis=0)
    values = evaluate(current)

    baseline = tuple(map(
        int,
        current[int(np.argmax(values))],
    ))

    # Hessian nastavlja od kontrolnih zavrsnih tacaka.
    while True:
        moved = False

        for j, row in enumerate(current):
            candidates = guided_pairs(
                row, model
            )

            if not len(candidates):
                continue

            values = evaluate(candidates)
            best = int(np.argmax(values))

            # Prihvatanje iskljucivo prema TACNOM skoru.
            if (
                values[best]
                > evaluate(row)[0] + 1e-10
            ):
                current[j], count = climb(
                    candidates[best]
                )

                steps += count + 1
                moved = True

        current = np.unique(
            current, axis=0
        )

        if not moved:
            break

    values = evaluate(current)
    best = int(np.argmax(values))

    prediction = tuple(map(
        int, current[best]
    ))

    assert (
        values[best]
        >= evaluate(
            np.array(baseline)
        )[0] - 1e-9
    )

    return (
        prediction,
        float(values[best]),
        steps,
        baseline,
    )


def predict(p, param):
    return search(fit(p, param))


def run_file(path, validation=48, test=64):
    h = load(path)
    split = len(h) - test

    if (
        validation < 16
        or test < 16
        or split - validation < WINDOW + 3
    ):
        raise ValueError(
            "Nedovoljno podataka ili prekratka provera."
        )

    totals = np.zeros(
        len(PARAMS), dtype=int
    )

    # Parametar se bira pre testnog perioda.
    for t in range(
        split - validation, split
    ):
        p = prepare(h[:t])

        for j, param in enumerate(PARAMS):
            totals[j] += hits(
                predict(p, param)[0],
                h[t],
            )

    param = PARAMS[
        int(np.argmax(totals))
    ]

    joint = []
    single = []
    previous = []
    rows = []
    changed = 0

    # Obe pretrage koriste isti nauceni skor.
    # Parametar je zamrznut tokom testa.
    for t in range(split, len(h)):
        pred, _, _, baseline = predict(
            prepare(h[:t]),
            param,
        )

        joint.append(hits(pred, h[t]))
        single.append(hits(baseline, h[t]))
        previous.append(hits(h[t - 1], h[t]))

        rows.append(pred)
        changed += pred != baseline

    print("\nCSV:", path)

    print(
        f"Validacija={validation}; "
        f"odvojeni test={test}; "
        f"parametri={param}"
    )

    for label, values in (
        ("HESSIAN V2", joint),
        (
            "Kontrola: isti skor, zamene jednog broja",
            single,
        ),
        (
            "Kontrola: prethodno izvuceno kolo",
            previous,
        ),
    ):
        print(
            f"{label}: "
            f"prosek={np.mean(values):.4f}; "
            f"pogoci 0..7="
            f"{np.bincount(values, minlength=8).tolist()}"
        )

    shape = geometry(
        gaps(np.asarray(rows))
    )

    actual = geometry(
        gaps(h[split:])
    )

    print(
        "Nizovi najmanje 5: model/stvarna kola =",
        int(np.sum(shape[:, 0] >= 5)),
        int(np.sum(actual[:, 0] >= 5)),
    )

    print(
        "Unutrasnja praznina najmanje 20: model/stvarna kola =",
        int(np.sum(shape[:, 1] >= 20)),
        int(np.sum(actual[:, 1] >= 20)),
    )

    unique, counts = np.unique(
        np.asarray(rows),
        axis=0,
        return_counts=True,
    )

    forms, form_counts = np.unique(
        shape,
        axis=0,
        return_counts=True,
    )

    print(
        "Razlicitih izbora:",
        len(unique),
        "; najvece ponavljanje:",
        int(counts.max()),
    )

    print(
        "Razlicitih oblika:",
        len(forms),
        "; najcesci oblik:",
        forms[
            int(np.argmax(form_counts))
        ].astype(int).tolist(),
        "; broj pojavljivanja:",
        int(form_counts.max()),
    )

    print(
        f"Hessian promenio izbor u "
        f"{changed}/{test} testnih kola."
    )

    pred, score, steps, baseline = predict(
        prepare(h), param
    )

    replay = bool(
        np.any(
            np.all(h == pred, axis=1)
        )
    )

    final_shape = geometry(
        gaps(pred)
    )[0].astype(int)

    print(
        f"SLEDECI RED {len(h) + 1}:",
        ",".join(map(str, pred)),
    )

    print(
        f"Skor={score:.10f}; "
        f"koraci={steps}; "
        f"istorijsko ponavljanje={replay}"
    )

    print(
        "Oblik: najduzi niz, najveca unutrasnja praznina =",
        final_shape.tolist(),
    )

    print(
        "Kontrolni izbor:",
        ",".join(map(str, baseline)),
    )

    print(
        f"Opseg: {STARTS} polazista, "
        "sve zamene jednog broja, "
        f"parovi {PAIR_STEPS.tolist()}, "
        f"do {PAIR_KEEP} predloga po tacki."
    )

    print(
        "Skor nije verovatnoca pogotka; "
        "globalni maksimum nije potvrden."
    )

    return pred


def self_test():
    # Nezavisna funkcija sa poznatim izvodima.
    matrix = np.diag(
        np.arange(1., 8.)
    )

    matrix[0, 1] = .3
    matrix[1, 0] = .3

    linear = np.arange(7.) / 10.

    q = np.array((
        .1, .2, -.1, .3, -.2, .4, .1
    ))

    def quadratic(x):
        return (
            x @ linear
            + .5 * np.einsum(
                "ni,ij,nj->n",
                x,
                matrix,
                x,
            )
        )

    grad, hess = numerical_derivatives(
        quadratic, q
    )

    assert np.allclose(
        grad,
        linear + matrix @ q,
        atol=1e-8,
    )

    assert np.allclose(
        hess,
        matrix,
        atol=1e-7,
    )

    base = np.array((
        (1, 2, 3, 4, 5, 6, 7),
        (4, 12, 18, 28, 35, 36, 37),
        (33, 34, 35, 36, 37, 38, 39),
        (4, 8, 13, 18, 23, 28, 34),
    ))

    g = gaps(base)

    assert np.all(
        g.sum(axis=1) == 32
    )

    assert np.array_equal(
        np.cumsum(g[:, :7] + 1, axis=1),
        base,
    )

    h = np.tile(base, (60, 1))
    p = prepare(h)
    model = fit(p, PARAMS[1])

    assert np.all(
        scores(base, model) <= 1e-12
    )

    pred, score, _, baseline = predict(
        p, PARAMS[1]
    )

    assert pred == predict(
        prepare(h.copy()),
        PARAMS[1],
    )[0]

    assert (
        len(set(pred)) == 7
        and 1 <= min(pred)
        <= max(pred) <= 39
    )

    assert (
        score
        >= scores(
            np.array(baseline), model
        )[0] - 1e-9
    )

    assert (
        scores(
            replacements(np.array(pred)),
            model,
        ).max()
        <= score + 1e-9
    )

    pairs = guided_pairs(
        np.array(pred), model
    )

    assert (
        scores(
            pairs, model
        ).max(initial=-np.inf)
        <= score + 1e-9
    )

    q = g[-1, :7] + .2

    a, b = numerical_derivatives(
        lambda x: score_q(x, model),
        q,
    )

    c, d = numerical_derivatives(
        lambda x: score_q(x, model),
        q,
        FD_STEP / 2.,
    )

    assert np.allclose(
        a, c,
        rtol=1e-4,
        atol=1e-5,
    )

    assert np.allclose(
        b, d,
        rtol=1e-3,
        atol=1e-4,
    )

    future = h.copy()
    future[-1] = base[1]

    assert predict(
        prepare(h[:-1]),
        PARAMS[1],
    )[0] == predict(
        prepare(future[:-1]),
        PARAMS[1],
    )[0]

    print(
        "Self-test OK: Hessian, empirijski skor, "
        "pretraga i hronologija."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "csv",
        nargs="*",
        help="Podrazumevano sva tri nova CSV fajla.",
    )

    parser.add_argument(
        "--validation",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--test",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()

    seen = {}

    for path in args.csv or DEFAULTS:
        prediction = run_file(
            path,
            args.validation,
            args.test,
        )

        if prediction in seen:
            print(
                "ISTI IZLAZ ZA DVA FAJLA:",
                seen[prediction],
                "i",
                path,
            )

        seen[prediction] = path


if __name__ == "__main__":
    main()



"""
CSV: /data/loto7_4700_k81.csv
Validacija=48; odvojeni test=64; parametri=(0.5, 0.05)
HESSIAN V2: prosek=1.4375; pogoci 0..7=[10, 24, 23, 6, 1, 0, 0, 0]
Kontrola: isti skor, zamene jednog broja: prosek=1.3750; pogoci 0..7=[9, 28, 22, 4, 1, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.2344; pogoci 0..7=[13, 30, 15, 5, 1, 0, 0, 0]
Nizovi najmanje 5: model/stvarna kola = 0 0
Unutrasnja praznina najmanje 20: model/stvarna kola = 1 1
Razlicitih izbora: 64 ; najvece ponavljanje: 1
Razlicitih oblika: 23 ; najcesci oblik: [2, 11] ; broj pojavljivanja: 10
Hessian promenio izbor u 16/64 testnih kola.
SLEDECI RED 4701: 1,x,21,y,29,z,39
Skor=-3.4746495471; koraci=39; istorijsko ponavljanje=False
Oblik: najduzi niz, najveca unutrasnja praznina = [1, 13]
Kontrolni izbor: 2,6,19,25,28,35,37
Opseg: 8 polazista, sve zamene jednog broja, parovi [-4, -2, -1, 1, 2, 4], do 24 predloga po tacki.
Skor nije verovatnoca pogotka; globalni maksimum nije potvrden.



CSV: /data/loto7_4700_k81_loto_2972.csv
Validacija=48; odvojeni test=64; parametri=(0.75, 0.1)
HESSIAN V2: prosek=1.2812; pogoci 0..7=[12, 31, 13, 7, 1, 0, 0, 0]
Kontrola: isti skor, zamene jednog broja: prosek=1.3281; pogoci 0..7=[11, 30, 16, 5, 2, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.3281; pogoci 0..7=[10, 30, 17, 7, 0, 0, 0, 0]
Nizovi najmanje 5: model/stvarna kola = 0 0
Unutrasnja praznina najmanje 20: model/stvarna kola = 0 1
Razlicitih izbora: 64 ; najvece ponavljanje: 1
Razlicitih oblika: 25 ; najcesci oblik: [2, 9] ; broj pojavljivanja: 6
Hessian promenio izbor u 17/64 testnih kola.
SLEDECI RED 2973: 7,x,14,y,31,z,35
Skor=-4.2611049325; koraci=72; istorijsko ponavljanje=False
Oblik: najduzi niz, najveca unutrasnja praznina = [2, 10]
Kontrolni izbor: 8,10,15,21,32,33,36
Opseg: 8 polazista, sve zamene jednog broja, parovi [-4, -2, -1, 1, 2, 4], do 24 predloga po tacki.
Skor nije verovatnoca pogotka; globalni maksimum nije potvrden.



CSV: /data/loto7_4700_k81_loto_plus_1728.csv
Validacija=48; odvojeni test=64; parametri=(0.5, 0.05)
HESSIAN V2: prosek=1.0781; pogoci 0..7=[16, 29, 17, 2, 0, 0, 0, 0]
Kontrola: isti skor, zamene jednog broja: prosek=1.1406; pogoci 0..7=[17, 28, 16, 0, 2, 1, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.2969; pogoci 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Nizovi najmanje 5: model/stvarna kola = 0 0
Unutrasnja praznina najmanje 20: model/stvarna kola = 0 1
Razlicitih izbora: 64 ; najvece ponavljanje: 1
Razlicitih oblika: 24 ; najcesci oblik: [2, 8] ; broj pojavljivanja: 9
Hessian promenio izbor u 17/64 testnih kola.
SLEDECI RED 1729: 2,x,6,y,16,z,35
Skor=-3.4467923505; koraci=48; istorijsko ponavljanje=False
Oblik: najduzi niz, najveca unutrasnja praznina = [2, 15]
Kontrolni izbor: 3,6,7,16,17,33,36
Opseg: 8 polazista, sve zamene jednog broja, parovi [-4, -2, -1, 1, 2, 4], do 24 predloga po tacki.
Skor nije verovatnoca pogotka; globalni maksimum nije potvrden.
"""



"""
Polinomski skor je uklonjen. 
Na 64 testna kola svaki CSV dao je 64 različite sedmorke, bez nizova od pet uzastopnih brojeva i bez stalnog geometrijskog oblika. 

CSV	        Prosek pogodaka	Sledeća kombinacija
Zbirni 4700	1,4375	        1,x,21,y,29,z,39
Loto 2972	1,2812	        7,x,14,y,31,z,35
Plus 1728	1,0781	        2,x,6,y,16,z,35

Hessian pretraga poboljšala je rezultat na zbirnom fajlu, ali ne i na druga dva.
"""
