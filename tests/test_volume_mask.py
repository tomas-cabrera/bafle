import numpy as np

from ligo.skymap.postprocess.crossmatch import crossmatch
import ligo.skymap.moc as lsm_moc

from bafle.io import gwtc
from bafle.models import volumemask


def test_plot_volume_mask():
    # Calculate the probability volume mask
    volume_mask = volumemask.get_volume_mask_from_gweventname("GW190521", 0.9)

    # Print the first 5 rows of the volume mask
    print(volume_mask[~np.isnan(volume_mask["DISTMIN"])][-5:])

    # Plot the volume mask
    volumemask.plot_volume_mask(volume_mask)


def test_sample_volume_mask():
    # Define gweventnames
    gweventnames = [
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
    ]
    # Get volume masks
    volume_masks = {}
    for gweventname in gweventnames:
        volume_mask = volumemask.get_volume_mask_from_gweventname(gweventname, 0.9)
        volume_masks[gweventname] = volume_mask
    # Sample volume masks
    samples, _ = volumemask.sample_from_volume_masks(
        volume_masks.values(),
        10,
        rng=np.random.default_rng(12345),
    )
    print(samples)


def test_mcmc_integration():
    # Define gweventnames
    gweventnames = [
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
    ]
    # Get volume masks
    volume_masks = {}
    for gweventname in gweventnames:
        print("*" * 40)
        print(gweventname)
        # Integrate skymap with ligo.skymap
        _, skymap = gwtc.get_gwtc_skymap(gweventname)
        lsm_integral = crossmatch(skymap, contours=[0.9]).contour_vols[0]
        print("lsm_integral:", lsm_integral)
        # Get volume mask
        volume_mask = volumemask.get_volume_mask_from_gweventname(gweventname, 0.9)
        volume_masks[gweventname] = volume_mask
        # Perform wedge integral
        areas = lsm_moc.uniq2pixarea(volume_mask["UNIQ"])
        mask = ~np.isnan(volume_mask["DISTMIN"]) & ~np.isnan(volume_mask["DISTMAX"])
        wedge_integral = np.sum(
            areas[mask]
            * (volume_mask["DISTMAX"][mask] ** 3 - volume_mask["DISTMIN"][mask] ** 3)
            / 3
        )
        print("wedge_integral:", wedge_integral)
        print("wedge/lsm ratio:", wedge_integral / lsm_integral)
        continue
        for i in np.arange(5, 10):
            # Integrate volume masks with mcmc
            mcmc_integral = volumemask.integrate_volume_masks_mcmc(
                [volume_mask],
                n_samples=2**i,
                nproc=32,
                rng=np.random.default_rng(12345),
            )
            print(i)
            print("mcmc_integral:", mcmc_integral)
            print("mcmc/lsm ratio:", mcmc_integral / lsm_integral)


################################################################################

if __name__ == "__main__":
    # test_plot_volume_mask()
    # test_sample_volume_mask()
    test_mcmc_integration()
