import os
import os.path as pa
import astropy.units as u
import astropy_healpix as ah
import ligo.skymap.distance as lsm_dist
import ligo.skymap.moc as lsm_moc
import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from bafle.io import gwtc
import parmap


def convert_skymap_to_prob_volume_mask(moc_skymap, ci_prob):
    """Convert a flattened skymap to a probability volume mask.

    Parameters
    ----------
    moc_skymap : astropy.table.Table
        A moc HEALPix skymap (flattened is not feasible for computation).
        Format assumes the following columns: "UNIQ", "PROBDENSITY", "DISTMU", "DISTSIGMA", "DISTNORM",
        as per LVK convention.
    ci_prob : float
        The probability threshold for the volume mask.

    Returns
    -------
    astropy.table.Table
        The probability volume mask.
        The columns are : "UNIQ", "DISTMIN", "DISTMAX",
        recording the distance interval for the tile in the 90% volume.
        "DISTMIN" and "DISTMAX" are in Mpc, and are NaN if the tile is not in the 90% volume.
    """
    # Copy the skymap
    skymap_temp = moc_skymap.copy()

    # Calculate pixel areas
    pixareas = lsm_moc.uniq2pixarea(skymap_temp["UNIQ"])

    # Shorthands for the columns
    probdens = skymap_temp["PROBDENSITY"]
    probs = skymap_temp["PROBDENSITY"] * pixareas
    distmus = skymap_temp["DISTMU"]
    distsigmas = skymap_temp["DISTSIGMA"]
    distnorms = skymap_temp["DISTNORM"]

    ### Set up the distance grid
    n_r = 1000
    # Calculate the marginal distance mean
    distmean, _ = lsm_dist.parameters_to_marginal_moments(probs, distmus, distsigmas)
    # Calculate the maximum distance
    max_r = 6 * distmean
    # Calculate the distance step
    d_r = max_r / n_r
    # Define the distance grid
    r = d_r * np.arange(0, n_r)

    # Calculate volume of each voxel, defined as the region within the
    # HEALPix pixel and contained within the two centric spherical shells
    # with radii (r - d_r / 2) and (r + d_r / 2).
    dV = (np.square(r) + np.square(d_r) / 12) * d_r * pixareas.reshape(-1, 1)

    # Calculate probability within each voxel.
    dP = (
        np.exp(
            -0.5
            * np.square(
                (r.reshape(1, -1) - distmus.reshape(-1, 1)) / distsigmas.reshape(-1, 1)
            )
        )
        * (probdens * distnorms / (distsigmas * np.sqrt(2 * np.pi))).reshape(-1, 1)
        * dV
    )
    dP[np.isnan(dP)] = 0  # Suppress invalid values

    # Calculate probability densities
    dP_dV = dP / dV

    # Get voxel indices sorted by probability density
    voxel_indices_sorted = np.argsort(dP_dV.flatten())[::-1]

    # Get voxel indices in the ci_prob volume
    cum_prob = np.cumsum(dP.flatten()[voxel_indices_sorted])
    ci_voxels = voxel_indices_sorted[cum_prob < ci_prob]

    # Create the volume mask
    volume_mask = Table()
    volume_mask["UNIQ"] = skymap_temp["UNIQ"]
    volume_mask["DISTMIN"] = np.nan * np.ones(len(skymap_temp))
    volume_mask["DISTMAX"] = np.nan * np.ones(len(skymap_temp))
    # Get min and max ci voxels
    ci_voxels_min = np.min(ci_voxels)
    ci_voxels_max = np.max(ci_voxels)
    # Iterate over the HEALPix pixels
    for i in range(len(skymap_temp)):
        # Get the voxel index limits for the pixel
        voxel_indices_pixel_min = i * n_r
        voxel_indices_pixel_max = (i + 1) * n_r - 1
        # Skip if not between min and max ci voxels
        if (
            voxel_indices_pixel_max < ci_voxels_min
            or voxel_indices_pixel_min > ci_voxels_max
        ):
            continue
        # Check for voxel indices in the ci_prob volume
        mask = (ci_voxels >= voxel_indices_pixel_min) & (
            ci_voxels <= voxel_indices_pixel_max
        )
        # If there are voxels in the ci_prob volume
        if np.any(mask):
            # Get the voxel indices in the pixel
            ci_voxels_in_pixel = ci_voxels[mask]
            # Convert the voxel indices to distance indices
            ci_voxels_in_pixel = ci_voxels_in_pixel - i * n_r
            # Get the minimum and maximum distances
            volume_mask["DISTMIN"][i] = r[ci_voxels_in_pixel[0]]
            volume_mask["DISTMAX"][i] = r[ci_voxels_in_pixel[-1]]

    return volume_mask


def get_volume_mask_from_gweventname(gweventname, ci_prob, force_recalc=False):
    # Get skymap
    skymap_path, skymap = gwtc.get_gwtc_skymap(gweventname)
    # Make volume mask path
    filename = pa.basename(skymap_path)
    filename = filename.replace(".fits", f"_volume_mask_{ci_prob}.fits")
    vm_path = pa.join(pa.dirname(__file__), ".cache", "volume_masks", filename)
    os.makedirs(pa.dirname(vm_path), exist_ok=True)
    # Load if exists
    if not force_recalc and pa.exists(vm_path):
        return Table.read(vm_path)
    # Convert skymap to volume mask
    volume_mask = convert_skymap_to_prob_volume_mask(skymap, ci_prob)
    # Save volume mask
    volume_mask.write(vm_path, overwrite=True)
    return volume_mask


def plot_volume_mask(volume_mask):
    """Makes a 3D plot of the probability volume mask.

    Parameters
    ----------
    volume_mask : _type_
        _description_
    """
    # Import the necessary modules
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Create a 3D plot
    fig2d = plt.figure()
    ax2d = fig2d.add_subplot()
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection="3d")

    # Get the indices of the pixels in the volume mask
    indices = np.where(~np.isnan(volume_mask["DISTMIN"]))[0]

    # Iterate over the pixels
    for i in indices:
        # Get the minimum and maximum distances
        uniq = volume_mask["UNIQ"][i]
        distmin = volume_mask["DISTMIN"][i]
        distmax = volume_mask["DISTMAX"][i]
        ### Get the pixel coordinates from the UNIQ index
        # Get the level and ipix
        level, ipix = ah.uniq_to_level_ipix(uniq)
        # Get nside
        nside = 2**level
        # Get the pixel coordinates
        ra, dec = ah.healpix_to_lonlat(ipix, nside, order="nested")
        # Convert from ra, dec, dist to x, y, z
        x = np.array([distmin, distmax]) * np.cos(ra.rad) * np.cos(dec.rad)
        y = np.array([distmin, distmax]) * np.sin(ra.rad) * np.cos(dec.rad)
        z = np.array([distmin, distmax]) * np.sin(dec.rad)
        # Plot the pixels
        ax2d.plot(
            ra.deg,
            dec.deg,
            ls="",
            marker=".",
            color="k",
            markersize=1,
            alpha=uniq / np.max(volume_mask["UNIQ"]),
        )
        ax3d.plot(
            x[0],
            y[0],
            z[0],
            ls="",
            marker=".",
            color="b",
            markersize=1,
            alpha=uniq / np.max(volume_mask["UNIQ"]),
        )
        ax3d.plot(
            x[1],
            y[1],
            z[1],
            ls="",
            marker=".",
            color="r",
            markersize=1,
            alpha=uniq / np.max(volume_mask["UNIQ"]),
        )

    # Set the axis limits
    ax3d.set_xlim(-np.nanmax(volume_mask["DISTMAX"]), np.nanmax(volume_mask["DISTMAX"]))
    ax3d.set_ylim(-np.nanmax(volume_mask["DISTMAX"]), np.nanmax(volume_mask["DISTMAX"]))
    ax3d.set_zlim(-np.nanmax(volume_mask["DISTMAX"]), np.nanmax(volume_mask["DISTMAX"]))

    # Set the axis labels
    ax2d.set_xlabel("RA (deg)")
    ax2d.set_ylabel("Dec (deg)")
    ax3d.set_xlabel("X (Mpc)")
    ax3d.set_ylabel("Y (Mpc)")
    ax3d.set_zlabel("Z (Mpc)")

    # Save the plot
    fig2d.savefig("volume_mask_2d.png")
    plt.close(fig=fig2d)
    fig3d.savefig("volume_mask.png")
    plt.close(fig=fig3d)


def sample_from_volume_masks(volume_masks, n_samples, dn_d3Mpc=None, rng=None):
    """Sample from a set of probability volume masks.

    Parameters
    ----------
    volume_masks : list of astropy.table.Table
        A list of probability volume masks.
        Each mask should have the columns "UNIQ", "DISTMIN", "DISTMAX".
    n_samples : int
        The number of samples to draw.
    dp_dMpc : float, optional
        The probability density at the distance of the samples.
        If None, the probability density is assumed to be uniform.

    Returns
    -------
    list of astropy.table.Table
        A list of samples.
        Each sample has the columns "DISTMU", "DISTSIGMA", "DISTNORM".
    """
    # Initialize the random number generator
    if rng is None:
        rng = np.random.default_rng()

    ### Distances
    # Define dn_d3Mpc distribution
    if dn_d3Mpc is None:

        def dn_d3Mpc(*args, **kwargs):
            return np.ones_like(args[0])

    ## Get min, max distances
    # Iterate over the volume masks
    distmin = np.inf
    distmax = 0
    for vm in volume_masks:
        # Get the minimum and maximum distances
        distmin = min(distmin, np.nanmin(vm["DISTMIN"]))
        distmax = max(distmax, np.nanmax(vm["DISTMAX"]))
    # Define the distance grid
    n_r = 1000
    r = np.linspace(distmin, distmax, n_r) * u.Mpc

    ### Sampling
    # Draw samples, avoiding double-counting overlapping volume masks
    p_dist = dn_d3Mpc(r) / np.sum(dn_d3Mpc(r))
    ras = []
    decs = []
    dists = []
    levels = np.arange(12)
    nsides = 2**levels
    ndraws = 0
    while len(ras) < n_samples:
        # Draw a random sample
        ra = rng.uniform(0, 2 * np.pi) * u.rad
        dec = np.arcsin(rng.uniform(-1, 1)) * u.rad
        dist = rng.choice(r, p=p_dist)
        ndraws += 1
        # Get uniqs for sample to level 12
        ipix = ah.lonlat_to_healpix(ra, dec, nsides, order="nested")
        uniqs = ah.level_ipix_to_uniq(levels, ipix)
        # Check if the sample is in the volume masks
        in_volume_masks = False
        for vm in volume_masks:
            # Check if the uniq is in the volume mask
            uniqs_in_vm = uniqs[np.isin(uniqs, vm["UNIQ"])]
            if len(uniqs_in_vm) == 0:
                continue
            uniq = uniqs_in_vm[0]
            uniq_i = np.where(vm["UNIQ"] == uniq)[0]
            # Check if the distance is in the volume mask
            if np.isnan(vm["DISTMIN"][uniq_i]) or np.isnan(vm["DISTMAX"][uniq_i]):
                continue
            if vm["DISTMIN"][uniq_i] <= dist <= vm["DISTMAX"][uniq_i]:
                in_volume_masks = True
                break
        # If the sample is in the volume masks
        if in_volume_masks:
            ras.append(ra.to(u.deg).value)
            decs.append(dec.to(u.deg).value)
            dists.append(dist)
    # Cast to astropy.table.Table
    samples = Table()
    samples["RA"] = ras * u.deg
    samples["DEC"] = decs * u.deg
    samples["DIST"] = dists * u.Mpc

    return samples, ndraws


def _sample_from_volume_masks(i, volume_masks, n_samples):
    return sample_from_volume_masks(
        volume_masks,
        n_samples,
        rng=np.random.default_rng(i),
    )


def integrate_volume_masks_mcmc(
    volume_masks,
    n_samples=1024,
    dn_d3Mpc=None,
    rng=None,
    nproc=1,
):
    # Calculate volume of spherical shell
    distmin = np.inf
    distmax = 0
    for vm in volume_masks:
        # Get the minimum and maximum distances
        distmin = min(distmin, np.nanmin(vm["DISTMIN"]))
        distmax = max(distmax, np.nanmax(vm["DISTMAX"]))
    # Calculate the volume of the spherical shell
    V = 4 / 3 * np.pi * (distmax**3 - distmin**3)
    # Sample from the volume masks
    if nproc > 1:
        n_samples_per_proc = n_samples // nproc
        n_samples = n_samples_per_proc * nproc
        result = parmap.map(
            _sample_from_volume_masks,
            rng.integers(0, 2**32, nproc),
            volume_masks,
            n_samples_per_proc,
            pm_processes=nproc,
        )
        ndraws = np.sum([s[1] for s in result])
        samples = np.concatenate([s[0] for s in result])
    else:
        samples, ndraws = sample_from_volume_masks(
            volume_masks,
            n_samples,
            rng=rng,
        )
    # Evaluate samples in the distance distribution
    if dn_d3Mpc is None:

        def dn_d3Mpc(*args, **kwargs):
            return np.ones_like(args[0])

    values = dn_d3Mpc(samples["DIST"])
    # Calculate the integral
    integral = np.sum(values) * V / ndraws
    return integral
