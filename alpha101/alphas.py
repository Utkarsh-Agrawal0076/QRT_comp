"""
The 101 Formulaic Alphas of Kakushadze (2015), arXiv:1601.00991.

Each `alphaNNN(d)` takes a `Data` panel (see data.py) exposing aligned
DataFrames (index = date, columns = ticker):

    d.open  d.high  d.low  d.close  d.volume  d.vwap  d.returns  d.cap
    d.adv(n)            -> average daily dollar volume over n days
    d.sector / d.industry / d.subindustry  -> ticker->group Series for indneutralize

and returns a DataFrame of the raw alpha signal (same shape). The signal is NOT
yet shifted for execution or scaled — the pipeline handles ranking / lagging /
neutralisation when it evaluates each alpha.

Formulas are transcribed verbatim from Appendix A. The paper's `^` (power) is
written as `power(...)`, and two-argument `max`/`min` (elementwise) as
`emax`/`emin`. `IndClass.subindustry` falls back to `industry` when no
sub-industry classification is available (flagged in ALPHA_META).
"""

import numpy as np
import pandas as pd

from .operators import (
    rank, scale, delay, delta, correlation, covariance, decay_linear,
    signedpower, power, ts_rank, ts_min, ts_max, ts_argmax, ts_argmin,
    ts_sum, product, stddev, sign, log, indneutralize, iif, emax, emin,
)

# paper-faithful aliases (shadow builtins within this module only)
abs = np.abs            # noqa: A001
sum = ts_sum            # noqa: A001
min = ts_min            # noqa: A001  (paper's min(x, d) == ts_min)
max = ts_max            # noqa: A001  (paper's max(x, d) == ts_max)


# --------------------------------------------------------------------------- #
def alpha001(d):
    inner = iif(d.returns < 0, stddev(d.returns, 20), d.close)
    return rank(ts_argmax(signedpower(inner, 2.0), 5)) - 0.5


def alpha002(d):
    return -1 * correlation(rank(delta(log(d.volume), 2)),
                            rank((d.close - d.open) / d.open), 6)


def alpha003(d):
    return -1 * correlation(rank(d.open), rank(d.volume), 10)


def alpha004(d):
    return -1 * ts_rank(rank(d.low), 9)


def alpha005(d):
    return rank(d.open - (ts_sum(d.vwap, 10) / 10)) * (-1 * abs(rank(d.close - d.vwap)))


def alpha006(d):
    return -1 * correlation(d.open, d.volume, 10)


def alpha007(d):
    adv20 = d.adv(20)
    val = (-1 * ts_rank(abs(delta(d.close, 7)), 60)) * sign(delta(d.close, 7))
    return iif(adv20 < d.volume, val, -1.0 * pd.DataFrame(1, index=d.close.index, columns=d.close.columns))


def alpha008(d):
    a = ts_sum(d.open, 5) * ts_sum(d.returns, 5)
    return -1 * rank(a - delay(a, 10))


def alpha009(d):
    dc = delta(d.close, 1)
    cond1 = 0 < ts_min(dc, 5)
    cond2 = ts_max(dc, 5) < 0
    return iif(cond1, dc, iif(cond2, dc, -1 * dc))


def alpha010(d):
    dc = delta(d.close, 1)
    cond1 = 0 < ts_min(dc, 4)
    cond2 = ts_max(dc, 4) < 0
    return rank(iif(cond1, dc, iif(cond2, dc, -1 * dc)))


def alpha011(d):
    vc = d.vwap - d.close
    return (rank(ts_max(vc, 3)) + rank(ts_min(vc, 3))) * rank(delta(d.volume, 3))


def alpha012(d):
    return sign(delta(d.volume, 1)) * (-1 * delta(d.close, 1))


def alpha013(d):
    return -1 * rank(covariance(rank(d.close), rank(d.volume), 5))


def alpha014(d):
    return (-1 * rank(delta(d.returns, 3))) * correlation(d.open, d.volume, 10)


def alpha015(d):
    return -1 * ts_sum(rank(correlation(rank(d.high), rank(d.volume), 3)), 3)


def alpha016(d):
    return -1 * rank(covariance(rank(d.high), rank(d.volume), 5))


def alpha017(d):
    adv20 = d.adv(20)
    return ((-1 * rank(ts_rank(d.close, 10))) *
            rank(delta(delta(d.close, 1), 1)) *
            rank(ts_rank(d.volume / adv20, 5)))


def alpha018(d):
    return -1 * rank(stddev(abs(d.close - d.open), 5) + (d.close - d.open) +
                     correlation(d.close, d.open, 10))


def alpha019(d):
    return ((-1 * sign((d.close - delay(d.close, 7)) + delta(d.close, 7))) *
            (1 + rank(1 + ts_sum(d.returns, 250))))


def alpha020(d):
    return ((-1 * rank(d.open - delay(d.high, 1))) *
            rank(d.open - delay(d.close, 1)) *
            rank(d.open - delay(d.low, 1)))


def alpha021(d):
    adv20 = d.adv(20)
    sma8 = ts_sum(d.close, 8) / 8
    sma2 = ts_sum(d.close, 2) / 2
    std8 = stddev(d.close, 8)
    cond1 = (sma8 + std8) < sma2
    cond2 = sma2 < (sma8 - std8)
    cond3 = (d.volume / adv20) >= 1
    one = pd.DataFrame(1.0, index=d.close.index, columns=d.close.columns)
    return iif(cond1, -1 * one, iif(cond2, one, iif(cond3, one, -1 * one)))


def alpha022(d):
    return -1 * (delta(correlation(d.high, d.volume, 5), 5) * rank(stddev(d.close, 20)))


def alpha023(d):
    cond = (ts_sum(d.high, 20) / 20) < d.high
    zero = pd.DataFrame(0.0, index=d.close.index, columns=d.close.columns)
    return iif(cond, -1 * delta(d.high, 2), zero)


def alpha024(d):
    ratio = delta(ts_sum(d.close, 100) / 100, 100) / delay(d.close, 100)
    cond = ratio <= 0.05
    return iif(cond, -1 * (d.close - ts_min(d.close, 100)), -1 * delta(d.close, 3))


def alpha025(d):
    adv20 = d.adv(20)
    return rank(((-1 * d.returns) * adv20) * d.vwap * (d.high - d.close))


def alpha026(d):
    return -1 * ts_max(correlation(ts_rank(d.volume, 5), ts_rank(d.high, 5), 5), 3)


def alpha027(d):
    cond = 0.5 < rank(ts_sum(correlation(rank(d.volume), rank(d.vwap), 6), 2) / 2.0)
    one = pd.DataFrame(1.0, index=d.close.index, columns=d.close.columns)
    return iif(cond, -1 * one, one)


def alpha028(d):
    adv20 = d.adv(20)
    return scale(correlation(adv20, d.low, 5) + ((d.high + d.low) / 2) - d.close)


def alpha029(d):
    inner = ts_min(
        product(rank(rank(scale(log(ts_sum(
            ts_min(rank(rank(-1 * rank(delta(d.close - 1, 5)))), 2), 1))))), 1), 5)
    return inner + ts_rank(delay(-1 * d.returns, 6), 5)


def alpha030(d):
    s = (sign(d.close - delay(d.close, 1)) +
         sign(delay(d.close, 1) - delay(d.close, 2)) +
         sign(delay(d.close, 2) - delay(d.close, 3)))
    return ((1.0 - rank(s)) * ts_sum(d.volume, 5)) / ts_sum(d.volume, 20)


def alpha031(d):
    adv20 = d.adv(20)
    return ((rank(rank(rank(decay_linear(-1 * rank(rank(delta(d.close, 10))), 10)))) +
             rank(-1 * delta(d.close, 3))) +
            sign(scale(correlation(adv20, d.low, 12))))


def alpha032(d):
    return (scale((ts_sum(d.close, 7) / 7) - d.close) +
            (20 * scale(correlation(d.vwap, delay(d.close, 5), 230))))


def alpha033(d):
    return rank(-1 * power(1 - (d.open / d.close), 1))


def alpha034(d):
    return rank((1 - rank(stddev(d.returns, 2) / stddev(d.returns, 5))) +
                (1 - rank(delta(d.close, 1))))


def alpha035(d):
    return (ts_rank(d.volume, 32) *
            (1 - ts_rank((d.close + d.high) - d.low, 16)) *
            (1 - ts_rank(d.returns, 32)))


def alpha036(d):
    adv20 = d.adv(20)
    return ((2.21 * rank(correlation(d.close - d.open, delay(d.volume, 1), 15))) +
            (0.7 * rank(d.open - d.close)) +
            (0.73 * rank(ts_rank(delay(-1 * d.returns, 6), 5))) +
            rank(abs(correlation(d.vwap, adv20, 6))) +
            (0.6 * rank(((ts_sum(d.close, 200) / 200) - d.open) * (d.close - d.open))))


def alpha037(d):
    return (rank(correlation(delay(d.open - d.close, 1), d.close, 200)) +
            rank(d.open - d.close))


def alpha038(d):
    return (-1 * rank(ts_rank(d.close, 10))) * rank(d.close / d.open)


def alpha039(d):
    adv20 = d.adv(20)
    return ((-1 * rank(delta(d.close, 7) * (1 - rank(decay_linear(d.volume / adv20, 9))))) *
            (1 + rank(ts_sum(d.returns, 250))))


def alpha040(d):
    return (-1 * rank(stddev(d.high, 10))) * correlation(d.high, d.volume, 10)


def alpha041(d):
    return power(d.high * d.low, 0.5) - d.vwap


def alpha042(d):
    return rank(d.vwap - d.close) / rank(d.vwap + d.close)


def alpha043(d):
    adv20 = d.adv(20)
    return ts_rank(d.volume / adv20, 20) * ts_rank(-1 * delta(d.close, 7), 8)


def alpha044(d):
    return -1 * correlation(d.high, rank(d.volume), 5)


def alpha045(d):
    return -1 * (rank(ts_sum(delay(d.close, 5), 20) / 20) *
                 correlation(d.close, d.volume, 2) *
                 rank(correlation(ts_sum(d.close, 5), ts_sum(d.close, 20), 2)))


def alpha046(d):
    diff = ((delay(d.close, 20) - delay(d.close, 10)) / 10) - ((delay(d.close, 10) - d.close) / 10)
    one = pd.DataFrame(1.0, index=d.close.index, columns=d.close.columns)
    return iif(0.25 < diff, -1 * one,
               iif(diff < 0, one, (-1 * one) * (d.close - delay(d.close, 1))))


def alpha047(d):
    adv20 = d.adv(20)
    return ((((rank(1 / d.close) * d.volume) / adv20) *
             ((d.high * rank(d.high - d.close)) / (ts_sum(d.high, 5) / 5))) -
            rank(d.vwap - delay(d.vwap, 5)))


def alpha048(d):
    num = (correlation(delta(d.close, 1), delta(delay(d.close, 1), 1), 250) *
           delta(d.close, 1)) / d.close
    num = indneutralize(num, d.subindustry)
    denom = ts_sum(power(delta(d.close, 1) / delay(d.close, 1), 2), 250)
    return num / denom


def alpha049(d):
    diff = ((delay(d.close, 20) - delay(d.close, 10)) / 10) - ((delay(d.close, 10) - d.close) / 10)
    one = pd.DataFrame(1.0, index=d.close.index, columns=d.close.columns)
    return iif(diff < (-1 * 0.1), one, (-1 * one) * (d.close - delay(d.close, 1)))


def alpha050(d):
    return -1 * ts_max(rank(correlation(rank(d.volume), rank(d.vwap), 5)), 5)


def alpha051(d):
    diff = ((delay(d.close, 20) - delay(d.close, 10)) / 10) - ((delay(d.close, 10) - d.close) / 10)
    one = pd.DataFrame(1.0, index=d.close.index, columns=d.close.columns)
    return iif(diff < (-1 * 0.05), one, (-1 * one) * (d.close - delay(d.close, 1)))


def alpha052(d):
    return ((((-1 * ts_min(d.low, 5)) + delay(ts_min(d.low, 5), 5)) *
             rank((ts_sum(d.returns, 240) - ts_sum(d.returns, 20)) / 220)) *
            ts_rank(d.volume, 5))


def alpha053(d):
    return -1 * delta(((d.close - d.low) - (d.high - d.close)) / (d.close - d.low), 9)


def alpha054(d):
    return ((-1 * ((d.low - d.close) * power(d.open, 5))) /
            ((d.low - d.high) * power(d.close, 5)))


def alpha055(d):
    inner = (d.close - ts_min(d.low, 12)) / (ts_max(d.high, 12) - ts_min(d.low, 12))
    return -1 * correlation(rank(inner), rank(d.volume), 6)


def alpha056(d):
    return 0 - (1 * (rank(ts_sum(d.returns, 10) / ts_sum(ts_sum(d.returns, 2), 3)) *
                     rank(d.returns * d.cap)))


def alpha057(d):
    return 0 - (1 * ((d.close - d.vwap) /
                     decay_linear(rank(ts_argmax(d.close, 30)), 2)))


def alpha058(d):
    return -1 * ts_rank(decay_linear(
        correlation(indneutralize(d.vwap, d.sector), d.volume, 3.92795), 7.89291), 5.50322)


def alpha059(d):
    inner = (d.vwap * 0.728317) + (d.vwap * (1 - 0.728317))
    return -1 * ts_rank(decay_linear(
        correlation(indneutralize(inner, d.industry), d.volume, 4.25197), 16.2289), 8.19648)


def alpha060(d):
    inner = (((d.close - d.low) - (d.high - d.close)) / (d.high - d.low)) * d.volume
    return 0 - (1 * ((2 * scale(rank(inner))) - scale(rank(ts_argmax(d.close, 10)))))


def alpha061(d):
    adv180 = d.adv(180)
    return (rank(d.vwap - ts_min(d.vwap, 16.1219)) <
            rank(correlation(d.vwap, adv180, 17.9282)))


def alpha062(d):
    adv20 = d.adv(20)
    cond = (rank(correlation(d.vwap, ts_sum(adv20, 22.4101), 9.91009)) <
            rank((rank(d.open) + rank(d.open)) <
                 (rank((d.high + d.low) / 2) + rank(d.high))))
    return cond * -1


def alpha063(d):
    adv180 = d.adv(180)
    inner = (d.vwap * 0.318108) + (d.open * (1 - 0.318108))
    return ((rank(decay_linear(delta(indneutralize(d.close, d.industry), 2.25164), 8.22237)) -
             rank(decay_linear(correlation(inner, ts_sum(adv180, 37.2467), 13.557), 12.2883))) * -1)


def alpha064(d):
    adv120 = d.adv(120)
    a = ts_sum((d.open * 0.178404) + (d.low * (1 - 0.178404)), 12.7054)
    b = ts_sum(adv120, 12.7054)
    c = delta((((d.high + d.low) / 2) * 0.178404) + (d.vwap * (1 - 0.178404)), 3.69741)
    return (rank(correlation(a, b, 16.6208)) < rank(c)) * -1


def alpha065(d):
    adv60 = d.adv(60)
    a = (d.open * 0.00817205) + (d.vwap * (1 - 0.00817205))
    return (rank(correlation(a, ts_sum(adv60, 8.6911), 6.40374)) <
            rank(d.open - ts_min(d.open, 13.635))) * -1


def alpha066(d):
    a = rank(decay_linear(delta(d.vwap, 3.51013), 7.23052))
    inner = (((d.low * 0.96633) + (d.low * (1 - 0.96633))) - d.vwap) / (d.open - ((d.high + d.low) / 2))
    b = ts_rank(decay_linear(inner, 11.4157), 6.72611)
    return (a + b) * -1


def alpha067(d):
    adv20 = d.adv(20)
    a = rank(d.high - ts_min(d.high, 2.14593))
    b = rank(correlation(indneutralize(d.vwap, d.sector),
                         indneutralize(adv20, d.subindustry), 6.02936))
    return power(a, b) * -1


def alpha068(d):
    adv15 = d.adv(15)
    a = ts_rank(correlation(rank(d.high), rank(adv15), 8.91644), 13.9333)
    b = rank(delta((d.close * 0.518371) + (d.low * (1 - 0.518371)), 1.06157))
    return (a < b) * -1


def alpha069(d):
    adv20 = d.adv(20)
    a = rank(ts_max(delta(indneutralize(d.vwap, d.industry), 2.72412), 4.79344))
    b = ts_rank(correlation((d.close * 0.490655) + (d.vwap * (1 - 0.490655)), adv20, 4.92416), 9.0615)
    return power(a, b) * -1


def alpha070(d):
    adv50 = d.adv(50)
    a = rank(delta(d.vwap, 1.29456))
    b = ts_rank(correlation(indneutralize(d.close, d.industry), adv50, 17.8256), 17.9171)
    return power(a, b) * -1


def alpha071(d):
    adv180 = d.adv(180)
    a = ts_rank(decay_linear(correlation(ts_rank(d.close, 3.43976),
                                         ts_rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948)
    b = ts_rank(decay_linear(power(rank((d.low + d.open) - (d.vwap + d.vwap)), 2), 16.4662), 4.4388)
    return emax(a, b)


def alpha072(d):
    adv40 = d.adv(40)
    a = rank(decay_linear(correlation((d.high + d.low) / 2, adv40, 8.93345), 10.1519))
    b = rank(decay_linear(correlation(ts_rank(d.vwap, 3.72469),
                                      ts_rank(d.volume, 18.5188), 6.86671), 2.95011))
    return a / b


def alpha073(d):
    a = rank(decay_linear(delta(d.vwap, 4.72775), 2.91864))
    inner = (d.open * 0.147155) + (d.low * (1 - 0.147155))
    b = ts_rank(decay_linear((delta(inner, 2.03608) / inner) * -1, 3.33829), 16.7411)
    return emax(a, b) * -1


def alpha074(d):
    adv30 = d.adv(30)
    a = rank(correlation(d.close, ts_sum(adv30, 37.4843), 15.1365))
    b = rank(correlation(rank((d.high * 0.0261661) + (d.vwap * (1 - 0.0261661))),
                         rank(d.volume), 11.4791))
    return (a < b) * -1


def alpha075(d):
    adv50 = d.adv(50)
    return (rank(correlation(d.vwap, d.volume, 4.24304)) <
            rank(correlation(rank(d.low), rank(adv50), 12.4413)))


def alpha076(d):
    adv81 = d.adv(81)
    a = rank(decay_linear(delta(d.vwap, 1.24383), 11.8259))
    b = ts_rank(decay_linear(ts_rank(correlation(indneutralize(d.low, d.sector),
                                                 adv81, 8.14941), 19.569), 17.1543), 19.383)
    return emax(a, b) * -1


def alpha077(d):
    adv40 = d.adv(40)
    a = rank(decay_linear((((d.high + d.low) / 2) + d.high) - (d.vwap + d.high), 20.0451))
    b = rank(decay_linear(correlation((d.high + d.low) / 2, adv40, 3.1614), 5.64125))
    return emin(a, b)


def alpha078(d):
    adv40 = d.adv(40)
    a = rank(correlation(ts_sum((d.low * 0.352233) + (d.vwap * (1 - 0.352233)), 19.7428),
                         ts_sum(adv40, 19.7428), 6.83313))
    b = rank(correlation(rank(d.vwap), rank(d.volume), 5.77492))
    return power(a, b)


def alpha079(d):
    adv150 = d.adv(150)
    a = rank(delta(indneutralize((d.close * 0.60733) + (d.open * (1 - 0.60733)), d.sector), 1.23438))
    b = rank(correlation(ts_rank(d.vwap, 3.60973), ts_rank(adv150, 9.18637), 14.6644))
    return (a < b)


def alpha080(d):
    adv10 = d.adv(10)
    a = rank(sign(delta(indneutralize((d.open * 0.868128) + (d.high * (1 - 0.868128)),
                                      d.industry), 4.04545)))
    b = ts_rank(correlation(d.high, adv10, 5.11456), 5.53756)
    return power(a, b) * -1


def alpha081(d):
    adv10 = d.adv(10)
    inner = rank(power(rank(correlation(d.vwap, ts_sum(adv10, 49.6054), 8.47743)), 4))
    a = rank(log(product(inner, 14.9655)))
    b = rank(correlation(rank(d.vwap), rank(d.volume), 5.07914))
    return (a < b) * -1


def alpha082(d):
    a = rank(decay_linear(delta(d.open, 1.46063), 14.8717))
    inner = (d.open * 0.634196) + (d.open * (1 - 0.634196))
    b = ts_rank(decay_linear(correlation(indneutralize(d.volume, d.sector), inner, 17.4842), 6.92131), 13.4283)
    return emin(a, b) * -1


def alpha083(d):
    hl = (d.high - d.low) / (ts_sum(d.close, 5) / 5)
    return ((rank(delay(hl, 2)) * rank(rank(d.volume))) / (hl / (d.vwap - d.close)))


def alpha084(d):
    return signedpower(ts_rank(d.vwap - ts_max(d.vwap, 15.3217), 20.7127), delta(d.close, 4.96796))


def alpha085(d):
    adv30 = d.adv(30)
    a = rank(correlation((d.high * 0.876703) + (d.close * (1 - 0.876703)), adv30, 9.61331))
    b = rank(correlation(ts_rank((d.high + d.low) / 2, 3.70596), ts_rank(d.volume, 10.1595), 7.11408))
    return power(a, b)


def alpha086(d):
    adv20 = d.adv(20)
    a = ts_rank(correlation(d.close, ts_sum(adv20, 14.7444), 6.00049), 20.4195)
    b = rank((d.open + d.close) - (d.vwap + d.open))
    return (a < b) * -1


def alpha087(d):
    adv81 = d.adv(81)
    a = rank(decay_linear(delta((d.close * 0.369701) + (d.vwap * (1 - 0.369701)), 1.91233), 2.65461))
    b = ts_rank(decay_linear(abs(correlation(indneutralize(adv81, d.industry), d.close, 13.4132)), 4.89768), 14.4535)
    return emax(a, b) * -1


def alpha088(d):
    adv60 = d.adv(60)
    a = rank(decay_linear((rank(d.open) + rank(d.low)) - (rank(d.high) + rank(d.close)), 8.06882))
    b = ts_rank(decay_linear(correlation(ts_rank(d.close, 8.44728),
                                         ts_rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957)
    return emin(a, b)


def alpha089(d):
    adv10 = d.adv(10)
    a = ts_rank(decay_linear(correlation((d.low * 0.967285) + (d.low * (1 - 0.967285)),
                                         adv10, 6.94279), 5.51607), 3.79744)
    b = ts_rank(decay_linear(delta(indneutralize(d.vwap, d.industry), 3.48158), 10.1466), 15.3012)
    return a - b


def alpha090(d):
    adv40 = d.adv(40)
    a = rank(d.close - ts_max(d.close, 4.66719))
    b = ts_rank(correlation(indneutralize(adv40, d.subindustry), d.low, 5.38375), 3.21856)
    return power(a, b) * -1


def alpha091(d):
    adv30 = d.adv(30)
    a = ts_rank(decay_linear(decay_linear(correlation(indneutralize(d.close, d.industry),
                                                      d.volume, 9.74928), 16.398), 3.83219), 4.8667)
    b = rank(decay_linear(correlation(d.vwap, adv30, 4.01303), 2.6809))
    return (a - b) * -1


def alpha092(d):
    adv30 = d.adv(30)
    a = ts_rank(decay_linear((((d.high + d.low) / 2) + d.close) < (d.low + d.open), 14.7221), 18.8683)
    b = ts_rank(decay_linear(correlation(rank(d.low), rank(adv30), 7.58555), 6.94024), 6.80584)
    return emin(a, b)


def alpha093(d):
    adv81 = d.adv(81)
    a = ts_rank(decay_linear(correlation(indneutralize(d.vwap, d.industry), adv81, 17.4193), 19.848), 7.54455)
    b = rank(decay_linear(delta((d.close * 0.524434) + (d.vwap * (1 - 0.524434)), 2.77377), 16.2664))
    return a / b


def alpha094(d):
    adv60 = d.adv(60)
    a = rank(d.vwap - ts_min(d.vwap, 11.5783))
    b = ts_rank(correlation(ts_rank(d.vwap, 19.6462), ts_rank(adv60, 4.02992), 18.0926), 2.70756)
    return power(a, b) * -1


def alpha095(d):
    adv40 = d.adv(40)
    a = rank(d.open - ts_min(d.open, 12.4105))
    inner = correlation(ts_sum((d.high + d.low) / 2, 19.1351), ts_sum(adv40, 19.1351), 12.8742)
    b = ts_rank(power(rank(inner), 5), 11.7584)
    return (a < b)


def alpha096(d):
    adv60 = d.adv(60)
    a = ts_rank(decay_linear(correlation(rank(d.vwap), rank(d.volume), 3.83878), 4.16783), 8.38151)
    inner = ts_argmax(correlation(ts_rank(d.close, 7.45404), ts_rank(adv60, 4.13242), 3.65459), 12.6556)
    b = ts_rank(decay_linear(inner, 14.0365), 13.4143)
    return emax(a, b) * -1


def alpha097(d):
    adv60 = d.adv(60)
    a = rank(decay_linear(delta(indneutralize((d.low * 0.721001) + (d.vwap * (1 - 0.721001)),
                                              d.industry), 3.3705), 20.4523))
    inner = correlation(ts_rank(d.low, 7.87871), ts_rank(adv60, 17.255), 4.97547)
    b = ts_rank(decay_linear(ts_rank(inner, 18.5925), 15.7152), 6.71659)
    return (a - b) * -1


def alpha098(d):
    adv5 = d.adv(5)
    adv15 = d.adv(15)
    a = rank(decay_linear(correlation(d.vwap, ts_sum(adv5, 26.4719), 4.58418), 7.18088))
    inner = ts_argmin(correlation(rank(d.open), rank(adv15), 20.8187), 8.62571)
    b = rank(decay_linear(ts_rank(inner, 6.95668), 8.07206))
    return a - b


def alpha099(d):
    adv60 = d.adv(60)
    a = rank(correlation(ts_sum((d.high + d.low) / 2, 19.8975), ts_sum(adv60, 19.8975), 8.8136))
    b = rank(correlation(d.low, d.volume, 6.28259))
    return (a < b) * -1


def alpha100(d):
    adv20 = d.adv(20)
    inner = rank(((((d.close - d.low) - (d.high - d.close)) / (d.high - d.low)) * d.volume))
    part1 = 1.5 * scale(indneutralize(indneutralize(inner, d.subindustry), d.subindustry))
    part2 = scale(indneutralize(correlation(d.close, rank(adv20), 5) - rank(ts_argmin(d.close, 30)),
                                d.subindustry))
    return 0 - (1 * ((part1 - part2) * (d.volume / adv20)))


def alpha101(d):
    return (d.close - d.open) / ((d.high - d.low) + 0.001)


# --------------------------------------------------------------------------- #
# Registry + metadata
# --------------------------------------------------------------------------- #
ALL_ALPHAS = {f"alpha{ i:03d}": globals()[f"alpha{i:03d}"] for i in range(1, 102)}

# alphas whose formula uses indneutralize (approximate when industry data is
# coarse / sub-industry is unavailable)
INDNEUTRAL_ALPHAS = {
    48, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100,
}

# alphas that reference market cap (uses a static cap proxy from the marketcap CSV)
CAP_ALPHAS = {56}


def get_alpha(n):
    """Return the alpha function for integer n (1..101)."""
    return ALL_ALPHAS[f"alpha{n:03d}"]
