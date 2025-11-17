
import numpy as np
import os


def parse_tfs_xml(xml_path):

    import xml.etree.ElementTree as ET
    # from collections import defaultdict

    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {
        "sem": "http://www.thermofisher.com/schemas/sem",
        "maps": "http://www.thermofisher.com/schemas/maps"
    }

    # --- Basic metadata ---

    data = {"Name": root.findtext("ImageMatrix/Name"), "Guid": root.findtext("ImageMatrix/Guid"),
            "TileWidth": float(root.findtext("ImageMatrix/TileWidth")),
            "TileHeight": float(root.findtext("ImageMatrix/TileHeight")),
            "TilePixelWidth": int(root.findtext("ImageMatrix/TilePixelWidth")),
            "TilePixelHeight": int(root.findtext("ImageMatrix/TilePixelHeight"))}

    data = dict(
        Name=root.findtext("ImageMatrix/Name"),
        Guid=root.findtext("ImageMatrix/Guid"),
        TileWidth=float(root.findtext("ImageMatrix/TileWidth")),
        TileHeight=float(root.findtext("ImageMatrix/TileHeight")),
        TilePixelWidth=int(root.findtext("ImageMatrix/TilePixelWidth")),
        TilePixelHeight=int(root.findtext("ImageMatrix/TilePixelHeight"))
    )

    # --- Channels ---
    channels = []
    for ch in root.findall("ImageMatrix/Channels/Channel"):
        channels.append({
            "Name": ch.findtext("Name"),
            "Guid": ch.findtext("Guid"),
            "Index": int(ch.findtext("Index")),
            "Color": {
                "A": int(ch.find("Color/A").text),
                "R": int(ch.find("Color/R").text),
                "G": int(ch.find("Color/G").text),
                "B": int(ch.find("Color/B").text),
            },
            "CameraBits": int(ch.findtext("CameraBits")),
            "Additive": ch.findtext("Additive") == "true"
        })
    data["Channels"] = channels

    # --- Images ---

    def safe_float(elem, tag):
        t = elem.findtext(tag) if elem is not None else None
        return float(t) if t not in (None, "", "NaN") else None

    images = []
    for img in root.findall("ImageMatrix/Images/Image"):
        idx = img.find("Index")
        pos = img.find("Position/sem:Position", ns)
        images.append({
            "Guid": img.findtext("Guid"),
            "Row": int(idx.findtext("Row")),
            "Column": int(idx.findtext("Column")),
            "Channel": int(idx.findtext("Channel")),
            "Plane": int(idx.findtext("Plane")),
            "TimeFrame": int(idx.findtext("TimeFrame")),
            "Position": {
                "X": float(pos.findtext("X")),
                "Y": float(pos.findtext("Y")),
                "Z": float(pos.findtext("Z")),
                "R": float(pos.findtext("R")),
                "AT": float(pos.findtext("AT")),
                "Focus": safe_float(pos, "Focus"),  # may be None
            },
            "RelativePath": img.findtext("RelativePath"),
            "Time": img.findtext("Time")
        })
    data["Images"] = images

    return data


def get_maximum_intensity_projections(parsed_data):
    """
    Returns a list of image entries corresponding to maximum intensity projections (MIPs).
    These are identified by 'MIP' in their RelativePath.
    """
    mip_images = [
        img for img in parsed_data.get("Images", [])
        if img.get("RelativePath") and "MIP" in img["RelativePath"]
    ]
    return mip_images


# def get_affine_transform(image_entry, px_size):
#     import math
#     from squirrel.library.affine_matrices import AffineMatrix
#     """
#     Returns a 3x3 2D affine transformation matrix based on an image's Position data.
#     Rotation is in degrees, translation in same units as X/Y (typically µm).
#     """
#     pos = image_entry.get("Position", {})
#     x = -pos.get("X", 0.0) * px_size
#     y = -pos.get("Y", 0.0) * px_size
#     r_deg = pos.get("R", 0.0)
#
#     # Convert degrees -> radians
#     r_rad = math.radians(r_deg)
#
#     cos_r = math.cos(r_rad)
#     sin_r = math.sin(r_rad)
#
#     # Build 3x3 homogeneous affine matrix
#     affine_matrix = np.array([
#         [cos_r, -sin_r, x],
#         [sin_r,  cos_r, y],
#         [0,      0,     1]
#     ])
#
#     return AffineMatrix(parameters=affine_matrix)


def get_affine_transform(image_entry, px_size, image_shape=None):
    import math
    import numpy as np
    from squirrel.library.affine_matrices import AffineMatrix
    """
    Returns a 3x3 2D affine transformation matrix based on an image's Position data.
    Rotation is in degrees, translation in same units as X/Y (typically µm).

    Parameters
    ----------
    image_entry : dict
        The image metadata entry (parsed from XML).
    px_size : float
        Pixel size in same units as X/Y position (e.g. µm per pixel).
    image_shape : tuple, optional
        (height, width) of the image. If given, rotation is performed about
        the image center instead of the origin.
    """
    pos = image_entry.get("Position", {})
    # x = -pos.get("X", 0.0) * px_size
    # y = -pos.get("Y", 0.0) * px_size
    x = 0
    y = 0
    r_deg = pos.get("R", 0.0) - 90 - 10

    # Convert degrees -> radians
    r_rad = math.radians(r_deg)
    cos_r = math.cos(r_rad)
    sin_r = math.sin(r_rad)

    # Base rotation + translation matrix
    R = np.array([
        [cos_r, -sin_r, 0],
        [sin_r,  cos_r, 0],
        [0,      0,     1]
    ])

    # Center-based rotation if image_shape provided
    if image_shape is not None:
        height, width = image_shape
        cx, cy = width / 2.0, height / 2.0

        # Translation matrices
        T1 = np.array([
            [1, 0, -cx],
            [0, 1, -cy],
            [0, 0,  1]
        ])
        T2 = np.array([
            [1, 0, cx],
            [0, 1, cy],
            [0, 0, 1]
        ])

        # Combine: translate to origin → rotate → translate back
        R = T2 @ R @ T1

    # Add stage translation (in world units, not pixels)
    R[0, 2] += x + pos.get("X", 0.0) * px_size
    R[1, 2] += y - pos.get("Y", 0.0) * px_size

    return AffineMatrix(parameters=R)


def get_image_filepath(image_entry, xml_path=None):
    """
    Returns the absolute file path for an image entry.

    Parameters
    ----------
    image_entry : dict
        A single image entry from the parsed XML (contains 'RelativePath').
    xml_path : str, optional
        Path to the XML file. If provided, the relative path is resolved against its directory.

    Returns
    -------
    str
        Absolute path to the image file.
    """
    rel_path = image_entry.get("RelativePath", "")
    if not rel_path:
        raise ValueError("Image entry has no 'RelativePath' field.")

    # Normalize Windows-style backslashes to system separators
    rel_path = rel_path.replace("\\", os.sep).replace("/", os.sep)

    if xml_path:
        base_dir = os.path.dirname(os.path.abspath(xml_path))
        abs_path = os.path.abspath(os.path.join(base_dir, rel_path))
    else:
        abs_path = os.path.abspath(rel_path)

    return abs_path


if __name__ == '__main__':

    fp = '/media/julian/Data/projects/hennies/cryo_mobie_devel/tomo_clem/2025-02-14_hrosa_Krios4_HR1c_MCC_pombe_starved/CLEM/Image_202502131321_stack_11.tfs.xml'
    d = parse_tfs_xml(xml_path=fp)

    mips = get_maximum_intensity_projections(d)

    affine = get_affine_transform(mips[0])

    pass
