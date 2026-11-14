"""Verify MIT licences omitted from six locked wheels' package metadata."""

from hashlib import sha256
from importlib.metadata import distribution

# The shipped LICENSE files were read in full; new wheels require a fresh review.
VERIFIED_MIT = {
    "conda-inject": ("1.3.2", "eb9920e8000c2dbd1ece0495215d62acd7094e2b40bac7c8eaff5e659d9827a1"),
    "snakemake-interface-common": ("1.23.0", "615738c532d632a19f14bdfad76d24eebcd7b6c4fa7b846426c87af15567d76b"),
    "snakemake-interface-logger-plugins": ("2.1.0", "b49355edba31771e7c3d412d3d0ade893f2b0dadb30a7387d2f59c036d3dfb05"),
    "snakemake-interface-report-plugins": ("1.3.0", "615738c532d632a19f14bdfad76d24eebcd7b6c4fa7b846426c87af15567d76b"),
    "snakemake-interface-scheduler-plugins": (
        "2.0.2",
        "b49355edba31771e7c3d412d3d0ade893f2b0dadb30a7387d2f59c036d3dfb05",
    ),
    "snakemake-interface-storage-plugins": (
        "4.4.1",
        "615738c532d632a19f14bdfad76d24eebcd7b6c4fa7b846426c87af15567d76b",
    ),
}


def check_metadata_gaps() -> None:
    """Refuse a new version or changed licence before licensecheck skips its metadata."""
    for name, (version, digest) in VERIFIED_MIT.items():
        package = distribution(name)
        license_file = next(file for file in package.files or [] if "LICENSE" in str(file))
        if package.version != version or sha256(package.locate_file(license_file).read_bytes()).hexdigest() != digest:
            msg = f"Review the shipped licence for {name} before updating its verified metadata gap"
            raise ValueError(msg)


if __name__ == "__main__":
    check_metadata_gaps()
