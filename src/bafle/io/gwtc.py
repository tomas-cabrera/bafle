import glob
import os
import os.path as pa

import astropy_healpix as ah
import ligo.skymap.moc as lsm_moc
from astropy.table import Table
from ligo.skymap.io import read_sky_map

################################################################################

MAPDIR = pa.join(pa.dirname(__file__), ".cache", "gwtc")

GWTC_PATHS = {
    "GWTC2": "https://dcc.ligo.org/public/0169/P2000223/007/all_skymaps.tar",
    "GWTC2.1": "https://zenodo.org/records/6513631/files/IGWN-GWTC2p1-v2-PESkyMaps.tar.gz",
    "GWTC3": "https://zenodo.org/records/8177023/files/IGWN-GWTC3p0-v2-PESkyLocalizations.tar.gz",
}


def _get_gwtc(mapdir):
    # Create directory
    os.makedirs(mapdir, exist_ok=True)

    # Download and extract GWTC skymaps
    for gwtc, path in GWTC_PATHS.items():
        # Download
        tarpath = pa.join(mapdir, f"{gwtc}.tar")
        if not pa.exists(tarpath):
            os.system(f"wget -O {tarpath} {path}")

        # Extract
        if not pa.exists(pa.join(mapdir, gwtc)):
            os.system(f"tar -xf {tarpath} -C {mapdir}")

            # Move GWTC2.1 skymaps to their own directory
            if gwtc == "GWTC2.1":
                gwtc2p1dir = pa.join(mapdir, "IGWN-GWTC2p1-v2-PESkyMaps")
                os.makedirs(gwtc2p1dir, exist_ok=True)
                os.system(f"mv {mapdir}/IGWN-GWTC2p1-v2-*.fits {gwtc2p1dir}")


def get_gwtc_skymap(
    gweventname,
    catdirs={
        "GWTC2": "all_skymaps",
        "GWTC2.1": "IGWN-GWTC2p1-v2-PESkyMaps",
        "GWTC3": "IGWN-GWTC3p0-v2-PESkyLocalizations",
    },
):
    # Choose GWTC, waveform
    ## If in GWTC2/GWTC2.1:
    if gweventname in [
        "GW190408_181802",
        "GW190412",
        "GW190413_052954",
        "GW190413_134308",
        "GW190421_213856",
        "GW190424_180648",
        "GW190425",
        "GW190426_152155",
        "GW190503_185404",
        "GW190512_180714",
        "GW190513_205428",
        "GW190514_065416",
        "GW190517_055101",
        "GW190519_153544",
        "GW190521",
        "GW190521_074359",
        "GW190527_092055",
        "GW190602_175927",
        "GW190620_030421",
        "GW190630_185205",
        "GW190701_203306",
        "GW190706_222641",
        "GW190707_093326",
        "GW190708_232457",
        "GW190719_215514",
        "GW190720_000836",
        "GW190727_060333",
        "GW190728_064510",
        "GW190731_140936",
        "GW190803_022701",
        "GW190814",
        "GW190828_063405",
        "GW190828_065509",
        "GW190909_114149",
        "GW190910_112807",
        "GW190915_235702",
        "GW190924_021846",
        "GW190929_012149",
        "GW190930_133541",
    ]:
        gwtc = "GWTC2"

        ### Select waveform
        if gweventname in [
            "GW190408_181802",
            "GW190412",
            "GW190426_152155",
            "GW190512_180714",
            "GW190707_093326",
            "GW190708_232457",
            "GW190720_000836",
            "GW190728_064510",
            "GW190814",
            "GW190828_065509",
            "GW190909_114149",
            "GW190910_112807",
            "GW190915_235702",
            "GW190924_021846",
            "GW190929_012149",
            "GW190930_133541",
        ]:
            waveform = "SEOBNRv4PHM"
        else:
            waveform = "NRSur7dq4"
    ## elif in GWTC2.1
    elif gweventname in [
        "GW190403_051519",
        "GW190426_190642",
        "GW190725_174728",
        "GW190805_211137",
        "GW190916_200658",
        "GW190917_114630",
        "GW190925_232845",
        "GW190926_050336",
    ]:
        gwtc = "GWTC2.1"
        waveform = "IMRPhenomXPHM"

    ## elif in GWTC3:
    elif gweventname in [
        "GW191103_012549",
        "GW191105_143521",
        "GW191109_010717",
        "GW191113_071753",
        "GW191126_115259",
        "GW191127_050227",
        "GW191129_134029",
        "GW191204_110529",
        "GW191204_171526",
        "GW191215_223052",
        "GW191216_213338",
        "GW191219_163120",
        "GW191222_033537",
        "GW191230_180458",
        "GW200105_162426",
        "GW200112_155838",
        "GW200115_042309",
        "GW200128_022011",
        "GW200129_065458",
        "GW200202_154313",
        "GW200208_130117",
        "GW200208_222617",
        "GW200209_085452",
        "GW200210_092254",
        "GW200216_220804",
        "GW200219_094415",
        "GW200220_061928",
        "GW200220_124850",
        "GW200224_222234",
        "GW200225_060421",
        "GW200302_015811",
        "GW200306_093714",
        "GW200308_173609",
        "GW200311_115853",
        "GW200316_215756",
        "GW200322_091133",
    ]:
        waveform = "IMRPhenomXPHM"
        gwtc = "GWTC3"
    else:
        raise ValueError(f"Event {gweventname} not found in GWTC2 or GWTC3")

    # Search catalogs from newest to oldest
    catdir = catdirs[gwtc]
    # glob for the skymap path
    ## Special case: GW190521 (GW190521_074359 also exists)
    if gweventname == "GW190521":
        globstr = f"{MAPDIR}/{catdir}/{gweventname}_C01:{waveform}.fits"
    ## Special case: GW190425 (BNS --> special waveform)
    elif gweventname == "GW190425":
        globstr = f"{MAPDIR}/{catdir}/{gweventname}_C01:SEOBNRv4T_surrogate_HS.fits"
    ## Special case: GW191219_163120, GW200115_042309 (NSBH --> high/low spin waveforms)
    elif gweventname in ["GW191219_163120", "GW200115_042309"]:
        globstr = f"{MAPDIR}/{catdir}/*{gweventname}*{waveform}:HighSpin.fits"
    else:
        globstr = f"{MAPDIR}/{catdir}/*{gweventname}*{waveform}*.fits"
    mappaths = glob.glob(globstr)

    # If multiple skymaps are found
    if len(mappaths) > 1:
        raise ValueError(f"Multiple skymaps found for {gweventname}")
    elif len(mappaths) == 0:
        raise ValueError(f"No skymaps found for {gweventname}")
    else:
        # Load skymap
        skymap_path = mappaths[0]
        hs = read_sky_map(skymap_path, moc=True)

    return skymap_path, hs


def get_flattened_skymap(gweventname):

    # Get skymap
    hs = get_gwtc_skymap(gweventname)

    # Flatten skymap
    hs_flat = Table(lsm_moc.rasterize(hs))
    hs_flat.meta = hs.meta

    # Calculate prob
    hs_flat["PROB"] = hs_flat["PROBDENSITY"] * ah.nside_to_pixel_area(
        ah.npix_to_nside(len(hs_flat))
    )

    return hs_flat


################################################################################

# Get skymaps upon import
if not pa.exists(MAPDIR):
    _get_gwtc(MAPDIR)
